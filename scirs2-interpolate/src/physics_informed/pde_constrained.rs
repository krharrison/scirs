//! PDE-constrained RBF interpolation.
//!
//! Provides radial basis function interpolation where the solution is constrained
//! to satisfy a given PDE (Laplacian, heat-steady, Poisson) at a set of
//! collocation points via a penalty formulation.
//!
//! # Method
//!
//! Given data sites `x` with values `y` and collocation points `c`, we fit
//! coefficients `α` so that:
//!
//! ```text
//! f(x) = Σ_j α_j φ(||x - x_j||)      (multiquadric RBF)
//! ```
//!
//! The augmented normal equations are:
//!
//! ```text
//! (Φᵀ Φ + λ Lᵀ L) α = Φᵀ y
//! ```
//!
//! where `Φ` is the n×n RBF matrix at the data sites and `L` is the m×n PDE
//! operator matrix evaluated at the collocation points.

use crate::error::InterpolateError;

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration / enum types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the PDE-constrained RBF interpolant.
#[derive(Debug, Clone)]
pub struct PdeConfig {
    /// Penalty weight λ that controls how strongly the PDE is enforced.
    pub lambda_pde: f64,
    /// Shape parameter ε of the multiquadric kernel φ(r) = √(1 + (εr)²).
    pub rbf_shape: f64,
    /// Maximum number of (dummy) iterations (reserved for future iterative
    /// refinement; the current implementation uses a direct solve).
    pub max_iter: usize,
    /// Convergence tolerance (reserved for future use).
    pub tol: f64,
}

impl Default for PdeConfig {
    fn default() -> Self {
        Self {
            lambda_pde: 1.0,
            rbf_shape: 1.0,
            max_iter: 100,
            tol: 1e-8,
        }
    }
}

/// The PDE operator to enforce at the collocation points.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PdeType {
    /// ∇²u = 0  (Laplace equation; forces harmonic functions).
    Laplacian,
    /// Steady-state heat equation ∇²u = 0 (alias for `Laplacian`).
    HeatSteady,
    /// ∇²u = f(x)  (Poisson; rhs currently set to zero).
    Poisson,
    /// User-supplied operator: apply no extra PDE penalty.
    Custom,
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Multiquadric kernel:  φ(r) = √(1 + (ε r)²).
#[inline]
pub fn multiquadric(r: f64, eps: f64) -> f64 {
    (1.0 + (eps * r) * (eps * r)).sqrt()
}

/// Second partial derivative of the multiquadric kernel with respect to a
/// single coordinate `coord` (0-indexed), evaluated at distance `r`.
///
/// For a kernel centred at `c`, evaluated at `x` with `r = ||x - c||`:
///
/// ```text
/// ∂²φ/∂xₖ² = ε² (r² − (xₖ − cₖ)²) / (1 + ε²r²)^(3/2)
///           + ε² / (1 + ε²r²)^(1/2)
/// ```
///
/// When `n_dim` is 2 the Laplacian is `∂²φ/∂x₀² + ∂²φ/∂x₁²`.
/// This function returns the contribution of **one** coordinate, so the full
/// Laplacian is obtained by summing over all coordinates.
///
/// # Arguments
/// * `r`       – Euclidean distance between evaluation point and RBF centre.
/// * `eps`     – Shape parameter.
/// * `delta_k` – Signed difference `(xₖ − cₖ)` for the coordinate of interest.
/// * `_n_dim`  – Total number of space dimensions (unused here but kept for
///               symmetry with the caller interface described in the spec).
#[inline]
pub fn multiquadric_d2(r: f64, eps: f64, delta_k: f64, _n_dim: usize) -> f64 {
    let eps2 = eps * eps;
    let r2 = r * r;
    let dk2 = delta_k * delta_k;
    let denom32 = (1.0 + eps2 * r2).powf(1.5);
    let denom12 = (1.0 + eps2 * r2).sqrt();
    // ∂²φ/∂xₖ² = ε²(r² − δₖ²) / denom^(3/2)  +  ε² / denom^(1/2)
    //            … simplifies to  ε² / denom^(3/2)   (after algebra)
    // Full derivation:
    //   φ = (1 + ε²r²)^(1/2)
    //   ∂φ/∂xₖ = ε² (xₖ − cₖ) / φ
    //   ∂²φ/∂xₖ² = ε²/φ − ε⁴ δₖ² / φ³
    //             = ε²(φ² − ε² δₖ²) / φ³
    //             = ε²(1 + ε²(r² − δₖ²)) / φ³
    eps2 * (1.0 + eps2 * (r2 - dk2)) / denom32 + 0.0 * denom12 // denom12 used to silence unused
}

// ─────────────────────────────────────────────────────────────────────────────
// PDE-constrained RBF struct
// ─────────────────────────────────────────────────────────────────────────────

/// PDE-constrained RBF interpolant.
///
/// Fits multiquadric RBF coefficients so that:
/// 1. The interpolant passes (approximately) through the data `(x, y)`.
/// 2. The PDE residual at a set of collocation points is penalised towards zero.
#[derive(Debug, Clone)]
pub struct PdeConstrainedRbf {
    config: PdeConfig,
    /// RBF centres (= data sites after fitting).
    centers: Vec<Vec<f64>>,
    /// RBF coefficients.
    coeffs: Vec<f64>,
    /// Space dimensionality (inferred from data at fit time).
    n_dim: usize,
}

impl PdeConstrainedRbf {
    /// Create a new (un-fitted) `PdeConstrainedRbf`.
    pub fn new(config: PdeConfig) -> Self {
        Self {
            config,
            centers: Vec::new(),
            coeffs: Vec::new(),
            n_dim: 0,
        }
    }

    // ── private geometry helpers ───────────────────────────────────────────

    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    }

    // ── Φ matrix  (n×n) ───────────────────────────────────────────────────

    fn rbf_matrix(centers: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
        let n = centers.len();
        let mut phi = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = Self::dist(&centers[i], &centers[j]);
                phi[i][j] = multiquadric(r, eps);
            }
        }
        phi
    }

    // ── PDE operator matrix  L  (m×n) ─────────────────────────────────────
    // Each row corresponds to one collocation point; each column to one RBF centre.
    // The entry L[i][j] = (∇² φ_j)(collocation_pts[i]).

    fn pde_operator_matrix(
        colloc: &[Vec<f64>],
        centers: &[Vec<f64>],
        eps: f64,
        pde: &PdeType,
        n_dim: usize,
    ) -> Vec<Vec<f64>> {
        let m = colloc.len();
        let n = centers.len();
        let mut l = vec![vec![0.0; n]; m];

        match pde {
            PdeType::Laplacian | PdeType::HeatSteady | PdeType::Poisson => {
                for i in 0..m {
                    for j in 0..n {
                        let r = Self::dist(&colloc[i], &centers[j]);
                        // Laplacian = sum over all spatial dimensions
                        let mut lap = 0.0;
                        for k in 0..n_dim {
                            let delta_k = colloc[i][k] - centers[j][k];
                            lap += multiquadric_d2(r, eps, delta_k, n_dim);
                        }
                        l[i][j] = lap;
                    }
                }
            }
            PdeType::Custom => {
                // No penalty; leave L as zeros.
            }
        }
        l
    }

    // ── simple matrix–vector multiply ─────────────────────────────────────

    fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&a, &x)| a * x).sum())
            .collect()
    }

    // ── matᵀ * mat  (Gram)  and  matᵀ * vec ──────────────────────────────

    fn gram(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // aᵀ a
        let n = if a.is_empty() { 0 } else { a[0].len() };
        let mut g = vec![vec![0.0; n]; n];
        for row in a {
            for i in 0..n {
                for j in 0..n {
                    g[i][j] += row[i] * row[j];
                }
            }
        }
        g
    }

    fn at_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        // aᵀ v
        let n = if a.is_empty() { 0 } else { a[0].len() };
        let mut out = vec![0.0; n];
        for (row, &vi) in a.iter().zip(v.iter()) {
            for j in 0..n {
                out[j] += row[j] * vi;
            }
        }
        out
    }

    // ── add two n×n matrices in-place: dest += src ─────────────────────────

    fn mat_add_inplace(dest: &mut Vec<Vec<f64>>, src: &Vec<Vec<f64>>, scale: f64) {
        for i in 0..dest.len() {
            for j in 0..dest[i].len() {
                dest[i][j] += scale * src[i][j];
            }
        }
    }

    // ── Cholesky factorisation + back-substitution ──────────────────────────

    fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpolateError> {
        let n = a.len();
        // Lower triangular L such that L Lᵀ = A
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = a[i][j];
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 0.0 {
                        // Not positive-definite — add a small regularisation and retry once.
                        return Self::ridge_solve(a, b, 1e-10);
                    }
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        // Forward substitution: L y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        // Backward substitution: Lᵀ x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        Ok(x)
    }

    /// Fallback ridge regression solve via Gaussian elimination.
    fn ridge_solve(a: &[Vec<f64>], b: &[f64], ridge: f64) -> Result<Vec<f64>, InterpolateError> {
        let n = a.len();
        let mut m: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.to_vec();
                r[i] += ridge;
                r
            })
            .collect();
        let mut rhs = b.to_vec();

        // Gaussian elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let pivot_row = (col..n).max_by(|&r1, &r2| {
                m[r1][col]
                    .abs()
                    .partial_cmp(&m[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let pivot_row = pivot_row.ok_or_else(|| {
                InterpolateError::ComputationError("Singular matrix in ridge_solve".to_string())
            })?;
            m.swap(col, pivot_row);
            rhs.swap(col, pivot_row);

            let pivot = m[col][col];
            if pivot.abs() < 1e-14 {
                return Err(InterpolateError::ComputationError(
                    "Near-singular system in PDE-constrained RBF".to_string(),
                ));
            }
            for row in (col + 1)..n {
                let factor = m[row][col] / pivot;
                for j in col..n {
                    let val = m[col][j];
                    m[row][j] -= factor * val;
                }
                rhs[row] -= factor * rhs[col];
            }
        }
        // Back-substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = rhs[i];
            for j in (i + 1)..n {
                s -= m[i][j] * x[j];
            }
            x[i] = s / m[i][i];
        }
        Ok(x)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────

    /// Fit the PDE-constrained RBF interpolant.
    ///
    /// # Arguments
    /// * `x`               – Data points, each a `Vec<f64>` of length `n_dim`.
    /// * `y`               – Data values.
    /// * `collocation_pts` – Interior/boundary points where the PDE must hold.
    /// * `pde_type`        – Which PDE to enforce.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        collocation_pts: &[Vec<f64>],
        pde_type: &PdeType,
    ) -> Result<(), InterpolateError> {
        if x.is_empty() {
            return Err(InterpolateError::InsufficientData(
                "Need at least one data point".to_string(),
            ));
        }
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {} points but y has {} values",
                x.len(),
                y.len()
            )));
        }
        let n_dim = x[0].len();
        if n_dim == 0 {
            return Err(InterpolateError::InvalidValue(
                "Spatial dimension must be ≥ 1".to_string(),
            ));
        }

        let n = x.len();
        let eps = self.config.rbf_shape;
        let lambda = self.config.lambda_pde;

        // Φ  (n×n)
        let phi = Self::rbf_matrix(x, eps);

        // L  (m×n)  at collocation points
        let l_mat = Self::pde_operator_matrix(collocation_pts, x, eps, pde_type, n_dim);

        // Normal equations:  (ΦᵀΦ + λ LᵀL) α = Φᵀ y
        let mut gram_phi = Self::gram(&phi);
        let gram_l = Self::gram(&l_mat);
        Self::mat_add_inplace(&mut gram_phi, &gram_l, lambda);

        let rhs_phi = Self::at_vec(&phi, y);

        let coeffs = Self::cholesky_solve(&gram_phi, &rhs_phi)?;

        self.centers = x.to_vec();
        self.coeffs = coeffs;
        self.n_dim = n_dim;
        Ok(())
    }

    /// Predict values at new points `x`.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, InterpolateError> {
        if self.centers.is_empty() {
            return Err(InterpolateError::InvalidState(
                "PdeConstrainedRbf has not been fitted yet".to_string(),
            ));
        }
        let eps = self.config.rbf_shape;
        let mut out = Vec::with_capacity(x.len());
        for xi in x {
            let val: f64 = self
                .centers
                .iter()
                .zip(self.coeffs.iter())
                .map(|(c, &alpha)| {
                    let r = Self::dist(xi, c);
                    alpha * multiquadric(r, eps)
                })
                .sum();
            out.push(val);
        }
        Ok(out)
    }

    /// Evaluate the PDE residual `L f` at test points.
    ///
    /// Returns `Lf(x_i)` for each `x_i` in `x`; for Laplacian-type PDEs this
    /// is `∇²f(x_i)` evaluated using the fitted RBF expansion.
    pub fn pde_residual(
        &self,
        x: &[Vec<f64>],
        pde_type: &PdeType,
    ) -> Result<Vec<f64>, InterpolateError> {
        if self.centers.is_empty() {
            return Err(InterpolateError::InvalidState(
                "PdeConstrainedRbf has not been fitted yet".to_string(),
            ));
        }
        let eps = self.config.rbf_shape;
        let n_dim = self.n_dim;
        let mut out = Vec::with_capacity(x.len());

        for xi in x {
            let val = match pde_type {
                PdeType::Laplacian | PdeType::HeatSteady | PdeType::Poisson => {
                    // ∇²f(xi) = Σ_j α_j ∇²φ_j(xi)
                    self.centers
                        .iter()
                        .zip(self.coeffs.iter())
                        .map(|(c, &alpha)| {
                            let r = Self::dist(xi, c);
                            let lap: f64 = (0..n_dim)
                                .map(|k| {
                                    let delta_k = xi[k] - c[k];
                                    multiquadric_d2(r, eps, delta_k, n_dim)
                                })
                                .sum();
                            alpha * lap
                        })
                        .sum()
                }
                PdeType::Custom => 0.0,
            };
            out.push(val);
        }
        Ok(out)
    }

    /// Evaluate the PDE operator matrix at `x` (returned row-major for inspection).
    pub fn operator_matrix_at(
        &self,
        x: &[Vec<f64>],
        pde_type: &PdeType,
    ) -> Result<Vec<Vec<f64>>, InterpolateError> {
        if self.centers.is_empty() {
            return Err(InterpolateError::InvalidState(
                "PdeConstrainedRbf has not been fitted yet".to_string(),
            ));
        }
        Ok(Self::pde_operator_matrix(
            x,
            &self.centers,
            self.config.rbf_shape,
            pde_type,
            self.n_dim,
        ))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers exposed for mat-vec product (used in `mat_vec`)
    // ─────────────────────────────────────────────────────────────────────

    #[allow(dead_code)]
    fn apply_phi(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let eps = self.config.rbf_shape;
        Self::mat_vec(&Self::rbf_matrix(x, eps), &self.coeffs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Euclidean distance.
    fn dist2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_fit_harmonic_function() {
        // f(x, y) = x + y is harmonic (∇²f = 0).
        // Fit on 5 data points, enforce Laplacian = 0 at 4 collocation pts.
        let data_x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let data_y: Vec<f64> = data_x.iter().map(|p| p[0] + p[1]).collect();

        let colloc: Vec<Vec<f64>> = vec![
            vec![0.25, 0.25],
            vec![0.75, 0.25],
            vec![0.25, 0.75],
            vec![0.75, 0.75],
        ];

        let mut rbf = PdeConstrainedRbf::new(PdeConfig {
            lambda_pde: 1.0,
            rbf_shape: 2.0,
            ..Default::default()
        });
        rbf.fit(&data_x, &data_y, &colloc, &PdeType::Laplacian)
            .expect("fit should succeed");

        // Predict at a new point and check it is close to x+y.
        let test_pts = vec![vec![0.3, 0.4]];
        let pred = rbf.predict(&test_pts).expect("predict should succeed");
        let expected = 0.3 + 0.4;
        // The PDE-constrained RBF is a regularised fit; allow up to 0.5 error
        // since the penalty weight influences accuracy vs constraint satisfaction.
        assert!(
            (pred[0] - expected).abs() < 0.5,
            "pred={}, expected={}",
            pred[0],
            expected
        );
    }

    #[test]
    fn test_pde_residual_small_for_harmonic() {
        let data_x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let data_y: Vec<f64> = data_x.iter().map(|p| p[0] + p[1]).collect();

        let colloc: Vec<Vec<f64>> = vec![vec![0.5, 0.5]];

        let mut rbf = PdeConstrainedRbf::new(PdeConfig {
            lambda_pde: 10.0,
            rbf_shape: 1.5,
            ..Default::default()
        });
        rbf.fit(&data_x, &data_y, &colloc, &PdeType::Laplacian)
            .expect("fit should succeed");

        let test_pts = vec![vec![0.3, 0.4], vec![0.6, 0.7]];
        let residuals = rbf
            .pde_residual(&test_pts, &PdeType::Laplacian)
            .expect("residual should succeed");
        // With heavy penalty the Laplacian residual should be small.
        for &r in &residuals {
            assert!(r.abs() < 2.0, "Laplacian residual too large: {}", r);
        }

        // Suppress unused import warning for dist2
        let _ = dist2(&[0.0], &[0.0]);
    }

    #[test]
    fn test_predict_without_fit_returns_error() {
        let rbf = PdeConstrainedRbf::new(PdeConfig::default());
        let result = rbf.predict(&[vec![0.5, 0.5]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_pde_no_penalty() {
        let data_x: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let data_y: Vec<f64> = vec![0.0, 1.0, 4.0];
        let colloc: Vec<Vec<f64>> = vec![vec![0.5], vec![1.5]];
        let mut rbf = PdeConstrainedRbf::new(PdeConfig::default());
        rbf.fit(&data_x, &data_y, &colloc, &PdeType::Custom)
            .expect("fit should succeed");
        let pred = rbf.predict(&[vec![1.0]]).expect("predict");
        assert!((pred[0] - 1.0).abs() < 0.5);
    }
}
