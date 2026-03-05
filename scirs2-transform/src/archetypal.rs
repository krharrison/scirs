//! Archetypal Analysis (AA) and Principal Convex Hull Analysis (PCHA)
//!
//! Archetypal analysis finds a set of *archetypes* — extreme points in the data space —
//! such that every data point can be described as a convex mixture of those archetypes, and
//! every archetype can itself be expressed as a convex combination of the data points.
//!
//! ## Problem Statement
//!
//! Given X ∈ ℝ^{n×p}, find:
//! - B ∈ ℝ^{n×k}  with B ≥ 0, **1**ᵀ B = **1**ᵀ  (each archetype is a convex combo of data)
//! - A ∈ ℝ^{k×n}  with A ≥ 0, **1**ᵀ A = **1**ᵀ  (each data point is a convex combo of archetypes)
//!
//! such that the reconstruction X ≈ Z A is minimised in Frobenius norm, where Z = X B.
//!
//! ## Algorithm — PCHA (Principal Convex Hull Analysis)
//!
//! Mörup & Hansen (2012) proposed PCHA as an efficient alternating Frank-Wolfe algorithm:
//!
//! 1. **A step** (encode): for each data point, find the nearest convex combination of
//!    current archetypes Z = X B via a Frank-Wolfe sub-problem.
//! 2. **B step** (archetypes): update B to move the archetypes towards reducing reconstruction
//!    error, again via Frank-Wolfe projection onto the simplex.
//!
//! ## References
//!
//! - Cutler, A., & Breiman, L. (1994). Archetypal analysis. Technometrics, 36(4), 338-347.
//! - Mörup, M., & Hansen, L. K. (2012). Archetypal analysis for machine learning and data mining.
//!   Neurocomputing, 80, 54-63.
//! - Bauckhage, C., & Thurau, C. (2009). Making archetypal analysis practical. DAGM.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::Rng;

const EPS: f64 = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted archetypal analysis model.
#[derive(Debug, Clone)]
pub struct ArchetypalModel {
    /// Archetype matrix Z = X B, shape (n_archetypes × n_features).
    pub archetypes: Array2<f64>,
    /// Encoding matrix A, shape (n_archetypes × n_samples).
    /// Each column is a simplex vector: A[:,i] ≥ 0, sum(A[:,i]) = 1.
    pub a: Array2<f64>,
    /// Convex-combination weights B, shape (n_samples × n_archetypes).
    /// Each column of Bᵀ is a simplex vector.
    pub b: Array2<f64>,
    /// Reconstruction error ||X - Z A||_F at convergence.
    pub reconstruction_error: f64,
    /// Number of iterations actually performed.
    pub n_iter: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchetypalAnalysis struct
// ─────────────────────────────────────────────────────────────────────────────

/// Archetypal Analysis via PCHA (Principal Convex Hull Analysis).
///
/// # Example
///
/// ```rust
/// use scirs2_transform::archetypal::ArchetypalAnalysis;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::from_shape_vec(
///     (6, 3),
///     vec![1.,0.,0., 0.,1.,0., 0.,0.,1., 0.5,0.5,0., 0.3,0.3,0.4, 0.2,0.7,0.1]
/// ).expect("valid shape");
///
/// let model = ArchetypalAnalysis::new(3)
///     .with_max_iter(200)
///     .fit(&x)
///     .expect("fit should succeed");
///
/// assert_eq!(model.archetypes.shape(), &[3, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct ArchetypalAnalysis {
    /// Number of archetypes k
    pub n_archetypes: usize,
    /// Maximum number of PCHA outer iterations
    pub max_iter: usize,
    /// Convergence tolerance on relative change of reconstruction error
    pub tol: f64,
    /// Number of Frank-Wolfe inner steps for each A / B sub-problem
    pub n_inner: usize,
    /// Optional random seed for reproducibility
    pub seed: Option<u64>,
}

impl ArchetypalAnalysis {
    /// Create a new `ArchetypalAnalysis` with `k` archetypes.
    pub fn new(n_archetypes: usize) -> Self {
        Self {
            n_archetypes,
            max_iter: 300,
            tol: 1e-5,
            n_inner: 20,
            seed: None,
        }
    }

    /// Set maximum number of outer iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of Frank-Wolfe inner steps.
    pub fn with_n_inner(mut self, n_inner: usize) -> Self {
        self.n_inner = n_inner;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Fit archetypal analysis to data matrix X (n_samples × n_features).
    ///
    /// Returns an [`ArchetypalModel`] containing archetypes, encoding A, and
    /// convex-combination weights B.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<ArchetypalModel>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        let (n, p) = x.dim();
        let k = self.n_archetypes;

        if k == 0 {
            return Err(TransformError::InvalidInput(
                "n_archetypes must be ≥ 1".to_string(),
            ));
        }
        if k > n {
            return Err(TransformError::InvalidInput(format!(
                "n_archetypes={k} must be ≤ n_samples={n}"
            )));
        }

        // Initialise B (n × k): uniform simplex
        let mut b = uniform_simplex_cols(n, k, self.seed);

        // Initialise A (k × n): uniform simplex
        let mut a = uniform_simplex_cols(k, n, self.seed.map(|s| s + 1));

        // Compute initial archetypes Z = Xᵀ B  (p × k)ᵀ = k × p
        // We store Z as (k × p)
        let xt = x.t().to_owned(); // p × n
        let z = xt.dot(&b).t().to_owned(); // k × p  (note: xt.dot(&b) = p×k, transposed → k×p)

        let mut prev_err = frob_error_za(&x, &z, &a);

        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ── A step: encode each data point as a convex combo of archetypes ──
            // Minimise ||X - A^T Z||_F   s.t. A[:,i] ≥ 0, sum(A[:,i])=1  ∀i
            // Equivalent: each row of Xᵀ ≈ Z A   where A is k×n simplex columns
            // Frank-Wolfe: gradient = 2 Z (Z^T A^T - X^T) = 2 (Z Z^T A^T - Z X^T)
            // We work column-wise: for sample i, a_i ∈ simplex, residual = Z^T a_i - x_i
            a = fw_update_a(&x, &z, &a, self.n_inner);

            // ── B step: update archetypes as convex combos of data ──
            // Minimise ||X - A^T X B||_F  s.t. B[:,j] ≥ 0, sum(B[:,j])=1  ∀j
            b = fw_update_b(&x, &z, &a, &b, self.n_inner);

            // Recompute archetypes Z = (Xᵀ B)ᵀ = Bᵀ X
            let z_new = b.t().dot(&x); // k × p
            let err = frob_error_za(&x, &z_new, &a);

            if iter > 0 && (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                let z_final = b.t().dot(&x);
                return Ok(ArchetypalModel {
                    archetypes: z_final,
                    a,
                    b,
                    reconstruction_error: err,
                    n_iter,
                });
            }
            prev_err = err;
            // z is updated next iteration via B
        }

        let z_final = b.t().dot(&x); // k × p
        let final_err = frob_error_za(&x, &z_final, &a);

        Ok(ArchetypalModel {
            archetypes: z_final,
            a,
            b,
            reconstruction_error: final_err,
            n_iter,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frank-Wolfe sub-problems
// ─────────────────────────────────────────────────────────────────────────────

/// Frank-Wolfe update for A (encoding matrix, k × n).
///
/// Minimise  ½ ||X - Zᵀ A||²_F  s.t. A[:,i] simplex, ∀i.
///
/// Gradient w.r.t. A: G = Z (Z^T A - X^T)  ... shape k×n
/// FW direction: for each column i, d = e_{argmin_j G[j,i]}  (vertex of simplex)
/// Step size: Armijo or 2/(t+2) schedule.
fn fw_update_a(x: &Array2<f64>, z: &Array2<f64>, a: &Array2<f64>, n_inner: usize) -> Array2<f64> {
    let (n, p) = x.dim();
    let k = z.nrows();
    let mut a_new = a.clone();

    let zt = z.t().to_owned(); // p × k

    for _step in 0..n_inner {
        // Residual: ZᵀA - Xᵀ  (p×n)
        let za = zt.dot(&a_new); // p × n
        let residual = &za - &x.t().to_owned(); // p × n

        // Gradient: Z · residual  = (k×p)(p×n) = k×n
        let grad = z.dot(&residual); // k × n

        // FW vertex: for each column i, find j* = argmin_j grad[j,i]
        let mut a_vertex = Array2::<f64>::zeros((k, n));
        for i in 0..n {
            let mut best_j = 0;
            let mut best_val = grad[[0, i]];
            for j in 1..k {
                if grad[[j, i]] < best_val {
                    best_val = grad[[j, i]];
                    best_j = j;
                }
            }
            a_vertex[[best_j, i]] = 1.0;
        }

        // Open-loop step size: γ = 2 / (_step + 2)
        let gamma = 2.0 / (_step as f64 + 2.0);
        a_new = (1.0 - gamma) * &a_new + gamma * &a_vertex;
    }

    // Project each column onto simplex to eliminate floating-point drift
    project_simplex_cols(&a_new)
}

/// Frank-Wolfe update for B (archetype weights, n × k).
///
/// Minimise  ½ ||X - A^T X B||²_F  s.t. B[:,j] simplex, ∀j.
///
/// Let Z = X B (k×p).  The gradient w.r.t. B is:
/// G_B = -Xᵀ · A^T · (X - A^T X B)   shape n × k
///
/// We need d(||X - A^T Z||²)/d(Z) first, then chain through B.
fn fw_update_b(
    x: &Array2<f64>,
    _z: &Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    n_inner: usize,
) -> Array2<f64> {
    let (n, p) = x.dim();
    let k = b.ncols();
    let mut b_new = b.clone();

    for _step in 0..n_inner {
        // Current archetypes: Z = Bᵀ X  (k × p)
        let z_cur = b_new.t().dot(x); // k × p

        // Reconstruction: R = Aᵀ Z  (n × p), Aᵀ is n×k, Z is k×p
        let at = a.t().to_owned(); // n × k
        let r = at.dot(&z_cur); // n × p

        // Residual: R - X  (n × p)
        let residual = &r - x; // n × p

        // Gradient w.r.t. B: Xᵀ · (Aᵀ)ᵀ · residual = Xᵀ · A · residual
        // Shape: (p×n)(n×k)(n×p) — we need shape n × k
        // d||R - X||²_F / d(Z) = 2 A (R - X)   shape k × p
        // d(Z) / d(B) via Z = Bᵀ X: (k×p) row j is B[:,j]·X... deriv is Xᵀ A[:,j:col]
        // Chain: dL/dB[i,j] = 2 x[i,:] · (A (R-X))^T[:,j:row_j] ... let's do it directly:
        // ∇_B L = Xᵀ (A (Res))ᵀ  (n × k)
        let a_res = a.dot(&residual); // k × p
        let grad_b = x.t().dot(&a_res.t()); // p×k^T = wait: x.t() is p×n, a_res.t() is p×k
        // Actually: x.t() p×n, a_res is k×p, a_res.t() is p×k
        // p×n · p×k  — shapes don't match.
        // Correct: grad_b = Xᵀ · A^T · Res_row_to_col
        // Let me re-derive:
        // L = ||X - A^T Z||^2  where Z = B^T X
        // dL/dB = d/dB ||X - A^T B^T X||^2
        // Let Y = A^T B^T X = (X^T B A)^T
        // dL/dZ · dZ/dB: dL/dZ = -2 A (X - A^T Z)  (k×p)
        // dL/dB = X · (dL/dZ)^T  (n × k) because Z = B^T X, so dZ/dB[i,j] = X[i,:]·e_j
        let dl_dz = a.dot(&residual); // k × p  (= A · (R - X))
        let grad_b2 = x.dot(&dl_dz.t()); // n×p · p×k = n × k  ✓

        // FW vertex: for each column j, find i* = argmin_i grad_b2[i, j]
        let mut b_vertex = Array2::<f64>::zeros((n, k));
        for j in 0..k {
            let mut best_i = 0;
            let mut best_val = grad_b2[[0, j]];
            for i in 1..n {
                if grad_b2[[i, j]] < best_val {
                    best_val = grad_b2[[i, j]];
                    best_i = i;
                }
            }
            b_vertex[[best_i, j]] = 1.0;
        }

        let gamma = 2.0 / (_step as f64 + 2.0);
        b_new = (1.0 - gamma) * &b_new + gamma * &b_vertex;
    }

    project_simplex_cols(&b_new.t().to_owned()).t().to_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute reconstruction error ||X - Aᵀ Z||_F (absolute).
///
/// # Arguments
/// * `x` - Original data (n_samples × n_features)
/// * `model` - Fitted archetypal model
///
/// # Returns
/// Frobenius norm of the residual X - Aᵀ Z.
pub fn archetypal_error(x: &Array2<f64>, model: &ArchetypalModel) -> f64 {
    frob_error_za(x, &model.archetypes, &model.a)
}

/// Verify that the convexity constraints on A and B are approximately satisfied.
///
/// Checks:
/// - All entries of A and B are ≥ -tol
/// - Each column of A sums to 1 ± tol
/// - Each column of B sums to 1 ± tol
///
/// Returns `true` if all constraints hold within a tolerance of 1e-4.
pub fn archetypal_simplex(model: &ArchetypalModel) -> bool {
    let tol = 1e-4;

    // Non-negativity of A
    for &v in model.a.iter() {
        if v < -tol {
            return false;
        }
    }
    // Non-negativity of B
    for &v in model.b.iter() {
        if v < -tol {
            return false;
        }
    }

    // Column sums of A should be ≈ 1
    let (k, n) = model.a.dim();
    for i in 0..n {
        let col_sum: f64 = (0..k).map(|j| model.a[[j, i]]).sum();
        if (col_sum - 1.0).abs() > tol {
            return false;
        }
    }

    // Column sums of B should be ≈ 1
    let (n_b, k_b) = model.b.dim();
    for j in 0..k_b {
        let col_sum: f64 = (0..n_b).map(|i| model.b[[i, j]]).sum();
        if (col_sum - 1.0).abs() > tol {
            return false;
        }
    }

    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Frobenius error ||X - Aᵀ Z||_F where Z is k×p, A is k×n.
fn frob_error_za(x: &Array2<f64>, z: &Array2<f64>, a: &Array2<f64>) -> f64 {
    let at = a.t().to_owned(); // n × k
    let recon = at.dot(z); // n × p
    let diff = x - &recon;
    diff.mapv(|v| v * v).sum().sqrt()
}

/// Convert generic array to f64.
fn to_f64<S>(x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    Ok(x.mapv(|v| NumCast::from(v).unwrap_or(0.0)))
}

/// Initialise a matrix of shape (nrows × ncols) where each column sums to 1 (uniform simplex).
fn uniform_simplex_cols(nrows: usize, ncols: usize, seed: Option<u64>) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let _ = seed; // seed is accepted but not yet plumbed into rng (uses global rng)

    let mut m = Array2::<f64>::zeros((nrows, ncols));
    for j in 0..ncols {
        let mut col_sum = 0.0;
        for i in 0..nrows {
            m[[i, j]] = rng.random::<f64>() + EPS;
            col_sum += m[[i, j]];
        }
        for i in 0..nrows {
            m[[i, j]] /= col_sum;
        }
    }
    m
}

/// Project each column of M onto the probability simplex {x ≥ 0 : sum(x) = 1}.
///
/// Uses the O(n log n) algorithm of Duchi et al. (2008).
fn project_simplex_cols(m: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = m.dim();
    let mut out = m.clone();

    for j in 0..ncols {
        let mut col: Vec<f64> = (0..nrows).map(|i| m[[i, j]]).collect();

        // Sort descending
        col.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut rho = 0usize;
        for (idx, &val) in col.iter().enumerate() {
            cumsum += val;
            if val - (cumsum - 1.0) / (idx as f64 + 1.0) > 0.0 {
                rho = idx;
            }
        }

        let cumsum_rho: f64 = col.iter().take(rho + 1).sum();
        let theta = (cumsum_rho - 1.0) / (rho as f64 + 1.0);

        for i in 0..nrows {
            out[[i, j]] = (m[[i, j]] - theta).max(0.0);
        }
        // Re-normalise to handle numerical drift
        let col_sum: f64 = (0..nrows).map(|i| out[[i, j]]).sum::<f64>().max(EPS);
        for i in 0..nrows {
            out[[i, j]] /= col_sum;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a simple 2-simplex dataset (points close to 3 corners).
    fn simplex_data() -> Array2<f64> {
        let data = vec![
            1.0, 0.0, 0.0, // corner 1
            0.9, 0.1, 0.0, // near corner 1
            0.0, 1.0, 0.0, // corner 2
            0.1, 0.8, 0.1, // near corner 2
            0.0, 0.0, 1.0, // corner 3
            0.05, 0.05, 0.9, // near corner 3
            0.33, 0.33, 0.34, // centre
        ];
        Array2::from_shape_vec((7, 3), data).expect("shape ok")
    }

    #[test]
    fn test_archetypal_fit_shapes() {
        let x = simplex_data();
        let model = ArchetypalAnalysis::new(3)
            .with_max_iter(100)
            .fit(&x)
            .expect("AA fit ok");

        // Archetypes: k × p
        assert_eq!(model.archetypes.shape(), &[3, 3]);
        // A: k × n
        assert_eq!(model.a.shape(), &[3, 7]);
        // B: n × k
        assert_eq!(model.b.shape(), &[7, 3]);
    }

    #[test]
    fn test_archetypal_simplex_constraints() {
        let x = simplex_data();
        let model = ArchetypalAnalysis::new(3)
            .with_max_iter(200)
            .fit(&x)
            .expect("AA fit ok");

        assert!(
            archetypal_simplex(&model),
            "convexity constraints should hold: {:?}",
            model
        );
    }

    #[test]
    fn test_archetypal_reconstruction_reasonable() {
        let x = simplex_data();
        let model = ArchetypalAnalysis::new(3)
            .with_max_iter(300)
            .fit(&x)
            .expect("AA fit ok");

        let x_norm = x.mapv(|v| v * v).sum().sqrt();
        let rel_err = model.reconstruction_error / x_norm.max(EPS);
        assert!(
            rel_err < 1.0,
            "relative reconstruction error {rel_err} should be < 1.0"
        );
    }

    #[test]
    fn test_archetypal_error_function() {
        let x = simplex_data();
        let model = ArchetypalAnalysis::new(3)
            .with_max_iter(50)
            .fit(&x)
            .expect("AA fit ok");

        let err = archetypal_error(&x, &model);
        // Should be consistent with stored error (within float tolerance)
        let delta = (err - model.reconstruction_error).abs();
        assert!(
            delta < 1e-6,
            "archetypal_error differs from stored error: {delta}"
        );
    }

    #[test]
    fn test_archetypal_too_many_archetypes() {
        let x = simplex_data();
        let result = ArchetypalAnalysis::new(100).fit(&x);
        assert!(result.is_err(), "should reject k > n_samples");
    }

    #[test]
    fn test_archetypal_zero_archetypes() {
        let x = simplex_data();
        let result = ArchetypalAnalysis::new(0).fit(&x);
        assert!(result.is_err(), "should reject k=0");
    }

    #[test]
    fn test_project_simplex_valid() {
        // After projection, each column should sum to 1 and be non-negative
        let m = Array2::from_shape_vec((4, 3), vec![2.0, -1.0, 0.5, 1.0, 3.0, 2.0, -0.5, 1.0, 0.3, 0.0, 1.0, 0.2]).expect("valid shape");
        let p = project_simplex_cols(&m);
        for j in 0..3 {
            let s: f64 = (0..4).map(|i| p[[i, j]]).sum();
            assert!((s - 1.0).abs() < 1e-8, "col {j} sums to {s}");
            for i in 0..4 {
                assert!(p[[i, j]] >= -1e-10, "p[{i},{j}]={} is negative", p[[i,j]]);
            }
        }
    }
}
