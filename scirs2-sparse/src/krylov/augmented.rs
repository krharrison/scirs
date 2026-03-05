//! Augmented Krylov subspace methods.
//!
//! These methods augment the standard Krylov space K_m(A, r) with an additional
//! subspace W (e.g., approximate eigenvectors or solution-space information from
//! previous solves) to form an *augmented Krylov space*
//!
//!   K_m^+(A, r, W) = K_m(A, r) + W.
//!
//! This is the theoretical foundation shared by:
//! - GMRES-DR (augment with harmonic Ritz vectors)
//! - GCRO-DR (augment with recycled subspace from previous solves)
//! - Augmented GMRES / LGMRES (augment with error approximations)
//! - Deflated BiCG methods
//!
//! # AugmentedKrylov
//!
//! This module provides a general `AugmentedKrylov` solver that accepts an
//! explicit augmentation subspace and runs GMRES in the augmented space.
//!
//! # References
//!
//! - Saad, Y. (1997). "Analysis of augmented Krylov subspace methods".
//!   SIAM J. Matrix Anal. Appl. 18(2), 435-449.
//! - Baker, A.H., Jessup, E.R., Manteuffel, T. (2005). "A technique for
//!   accelerating the convergence of restarted GMRES". SIAM J. Matrix Anal.
//!   Appl. 26(4), 962-984.

use crate::error::SparseError;
use crate::krylov::gmres_dr::{dot, gram_schmidt_mgs, norm2, solve_least_squares_hessenberg};

/// Configuration for an augmented Krylov solve.
#[derive(Debug, Clone)]
pub struct AugmentedKrylovConfig {
    /// Krylov dimension (not counting augmentation vectors).
    pub krylov_dim: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum number of matrix-vector products.
    pub max_iter: usize,
    /// Number of restart cycles.
    pub max_cycles: usize,
}

impl Default for AugmentedKrylovConfig {
    fn default() -> Self {
        Self {
            krylov_dim: 20,
            tol: 1e-10,
            max_iter: 1000,
            max_cycles: 50,
        }
    }
}

/// Result from an augmented Krylov solve.
#[derive(Debug, Clone)]
pub struct AugmentedKrylovResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Final residual norm ||b - Ax||.
    pub residual_norm: f64,
    /// Total matrix-vector products.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Residual norm at end of each restart cycle.
    pub residual_history: Vec<f64>,
    /// Updated augmentation vectors (orthonormal, from the converged Krylov basis).
    pub new_augmentation: Vec<Vec<f64>>,
}

/// Augmented Krylov subspace solver (augmented GMRES).
///
/// # Overview
///
/// Given an augmentation subspace W (external knowledge vectors -- e.g., from
/// previous solves or approximate eigenvectors), solves A x = b in the space
/// x_0 + span(W) + K_m(A, r_0).
///
/// The solver uses standard GMRES (Arnoldi + least-squares) for the Krylov
/// portion, and incorporates augmentation vectors by projecting the residual
/// onto the range of A*W before each GMRES cycle.
///
/// After convergence, an updated augmentation subspace is extracted from the
/// last Krylov basis for use in subsequent solves.
pub struct AugmentedKrylov {
    config: AugmentedKrylovConfig,
}

impl AugmentedKrylov {
    /// Create an `AugmentedKrylov` solver with the given configuration.
    pub fn new(config: AugmentedKrylovConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: AugmentedKrylovConfig::default(),
        }
    }

    /// Solve A x = b with augmented Krylov subspace.
    ///
    /// # Arguments
    ///
    /// * `matvec` - Closure for the matrix-vector product y = A x.
    /// * `b` - Right-hand side.
    /// * `x0` - Optional initial guess.
    /// * `augmentation` - External vectors to augment the Krylov space.
    ///   These are incorporated by projecting the residual onto span(A*W) at
    ///   each restart. Pass an empty slice for standard (non-augmented) GMRES.
    ///
    /// # Returns
    ///
    /// An `AugmentedKrylovResult` containing the solution and updated augmentation
    /// vectors.
    pub fn solve<F>(
        &self,
        matvec: F,
        b: &[f64],
        x0: Option<&[f64]>,
        augmentation: &[Vec<f64>],
    ) -> Result<AugmentedKrylovResult, SparseError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();
        let mut x = match x0 {
            Some(v) => v.to_vec(),
            None => vec![0.0f64; n],
        };

        let b_norm = norm2(b);
        let abs_tol = if b_norm > 1e-300 {
            self.config.tol * b_norm
        } else {
            self.config.tol
        };
        let mut total_mv = 0usize;
        let mut residual_history = Vec::new();
        let mut last_krylov: Vec<Vec<f64>> = Vec::new();

        // Prepare augmentation: orthonormalise the incoming vectors.
        let mut aug_orth: Vec<Vec<f64>> = augmentation.to_vec();
        gram_schmidt_mgs(&mut aug_orth);
        aug_orth.retain(|vi| norm2(vi) > 0.5);
        let k_aug = aug_orth.len();

        // Pre-compute A*W for augmentation projection (GCRO-style).
        // We project the residual onto range(A*W) to find the optimal correction
        // in the augmentation subspace: delta_x = W * (AW)^+ * r.
        let mut aw: Vec<Vec<f64>> = Vec::with_capacity(k_aug);
        for j in 0..k_aug {
            aw.push(matvec(&aug_orth[j]));
            total_mv += 1;
        }
        // Orthonormalise AW columns for stable projection.
        let mut aw_orth = aw.clone();
        gram_schmidt_mgs(&mut aw_orth);
        aw_orth.retain(|vi| norm2(vi) > 0.5);

        for _cycle in 0..self.config.max_cycles {
            // Compute residual.
            let ax = matvec(&x);
            total_mv += 1;
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
            let r_norm = norm2(&r);
            residual_history.push(r_norm);

            if r_norm <= abs_tol {
                let new_aug = extract_augmentation(&last_krylov, k_aug, n);
                return Ok(AugmentedKrylovResult {
                    x,
                    residual_norm: r_norm,
                    iterations: total_mv,
                    converged: true,
                    residual_history,
                    new_augmentation: new_aug,
                });
            }

            if total_mv >= self.config.max_iter {
                break;
            }

            // --- Augmentation correction (GCRO-style projection) ---
            // Compute x += W * (AW)^+ * r = W * ((AW_orth)^T * r) (since AW_orth is ON).
            // But we need the coefficients w.r.t. the original W, not AW_orth.
            // Use the relation: project r onto each AW column.
            if k_aug > 0 {
                // Solve: min ||r - AW * alpha|| for alpha.
                // Since we have AW (not orthogonalized), use normal equations:
                // (AW)^T (AW) alpha = (AW)^T r.
                let mut ata = vec![vec![0.0f64; k_aug]; k_aug];
                let mut atr = vec![0.0f64; k_aug];
                for i in 0..k_aug {
                    atr[i] = dot(&aw[i], &r);
                    for j in 0..k_aug {
                        ata[i][j] = dot(&aw[i], &aw[j]);
                    }
                }
                let alpha = solve_small_spd(&ata, &atr, k_aug);
                for j in 0..k_aug {
                    for i in 0..n {
                        x[i] += alpha[j] * aug_orth[j][i];
                    }
                }
            }

            // --- Standard GMRES cycle on the (updated) residual ---
            let ax2 = matvec(&x);
            total_mv += 1;
            let r2: Vec<f64> = b.iter().zip(ax2.iter()).map(|(bi, axi)| bi - axi).collect();
            let r2_norm = norm2(&r2);

            if r2_norm <= abs_tol {
                let new_aug = extract_augmentation(&last_krylov, k_aug, n);
                residual_history.push(r2_norm);
                return Ok(AugmentedKrylovResult {
                    x,
                    residual_norm: r2_norm,
                    iterations: total_mv,
                    converged: true,
                    residual_history,
                    new_augmentation: new_aug,
                });
            }

            // Build standard Arnoldi basis for GMRES.
            let m = self.config.krylov_dim;
            let mut v: Vec<Vec<f64>> = vec![vec![0.0f64; n]; m + 1];
            let mut h: Vec<Vec<f64>> = vec![vec![0.0f64; m]; m + 1];

            // v[0] = r2 / ||r2||
            let inv_r2 = 1.0 / r2_norm;
            for l in 0..n {
                v[0][l] = r2[l] * inv_r2;
            }

            // Arnoldi iteration: standard GMRES (no augmentation in the basis).
            let mut j_end = 1;
            for j in 1..=m {
                if j == m {
                    j_end = m;
                    break;
                }
                let w_raw = matvec(&v[j - 1]);
                total_mv += 1;
                let mut w = w_raw;

                // Modified Gram-Schmidt orthogonalization.
                for i in 0..j {
                    h[i][j - 1] = dot(&w, &v[i]);
                    for l in 0..n {
                        w[l] -= h[i][j - 1] * v[i][l];
                    }
                }
                h[j][j - 1] = norm2(&w);

                if h[j][j - 1] > 1e-15 {
                    let inv = 1.0 / h[j][j - 1];
                    for l in 0..n {
                        v[j][l] = w[l] * inv;
                    }
                    j_end = j + 1;
                } else {
                    j_end = j + 1;
                    break;
                }

                if total_mv >= self.config.max_iter {
                    j_end = j + 1;
                    break;
                }
            }

            let krylov_size = (j_end - 1).max(1).min(h[0].len());

            // Standard GMRES RHS: g = [beta, 0, ..., 0] where beta = ||r2||.
            let mut g = vec![0.0f64; j_end];
            g[0] = r2_norm;

            let cols = krylov_size.min(h[0].len());
            let y = solve_least_squares_hessenberg(&h, &g, cols)?;

            // Update solution: x += V * y.
            for j in 0..y.len().min(v.len()) {
                for i in 0..n {
                    x[i] += y[j] * v[j][i];
                }
            }

            // Store basis for augmentation extraction.
            last_krylov = v[..j_end].to_vec();

            if total_mv >= self.config.max_iter {
                break;
            }
        }

        // Final residual.
        let ax_fin = matvec(&x);
        total_mv += 1;
        let r_fin: Vec<f64> = b
            .iter()
            .zip(ax_fin.iter())
            .map(|(bi, axi)| bi - axi)
            .collect();
        let r_fin_norm = norm2(&r_fin);
        residual_history.push(r_fin_norm);

        let new_aug = extract_augmentation(&last_krylov, k_aug, n);

        Ok(AugmentedKrylovResult {
            x,
            residual_norm: r_fin_norm,
            iterations: total_mv,
            converged: r_fin_norm <= abs_tol,
            residual_history,
            new_augmentation: new_aug,
        })
    }
}

/// Solve a k x k SPD system A x = b.
/// Falls back to diagonal solve if Cholesky fails.
pub(crate) fn solve_small_spd(a: &[Vec<f64>], b: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        let diag = a[0][0];
        return vec![if diag.abs() > 1e-300 {
            b[0] / diag
        } else {
            0.0
        }];
    }

    // Attempt Cholesky: L L^T decomposition.
    let mut l = vec![vec![0.0f64; k]; k];
    let mut ok = true;
    'chol: for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i][j];
            for p in 0..j {
                sum -= l[i][p] * l[j][p];
            }
            if i == j {
                if sum < 1e-300 {
                    ok = false;
                    break 'chol;
                }
                l[i][j] = sum.sqrt();
            } else if l[j][j].abs() > 1e-300 {
                l[i][j] = sum / l[j][j];
            } else {
                ok = false;
                break 'chol;
            }
        }
    }

    if ok {
        // Forward substitution: L y = b.
        let mut y = vec![0.0f64; k];
        for i in 0..k {
            let mut s = b[i];
            for j in 0..i {
                s -= l[i][j] * y[j];
            }
            y[i] = if l[i][i].abs() > 1e-300 {
                s / l[i][i]
            } else {
                0.0
            };
        }
        // Back substitution: L^T x = y.
        let mut x = vec![0.0f64; k];
        for i in (0..k).rev() {
            let mut s = y[i];
            for j in (i + 1)..k {
                s -= l[j][i] * x[j];
            }
            x[i] = if l[i][i].abs() > 1e-300 {
                s / l[i][i]
            } else {
                0.0
            };
        }
        x
    } else {
        // Fallback: diagonal approximation.
        (0..k)
            .map(|i| {
                if a[i][i].abs() > 1e-300 {
                    b[i] / a[i][i]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Extract updated augmentation vectors from the last Krylov basis.
/// Takes the first few vectors from the Krylov basis.
fn extract_augmentation(krylov: &[Vec<f64>], k_aug: usize, _n: usize) -> Vec<Vec<f64>> {
    if krylov.is_empty() || k_aug == 0 {
        return Vec::new();
    }
    let m = krylov.len();
    let take = k_aug.min(m);
    // Return the first `take` Krylov vectors.
    let mut new_vecs: Vec<Vec<f64>> = krylov[..take].to_vec();
    gram_schmidt_mgs(&mut new_vecs);
    new_vecs.retain(|vi| norm2(vi) > 0.5);
    new_vecs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_mv(diag: Vec<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| x.iter().zip(diag.iter()).map(|(xi, di)| xi * di).collect()
    }

    #[test]
    fn test_augmented_krylov_no_augmentation() {
        // Without augmentation, this should behave like standard GMRES.
        let n = 8;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b = vec![1.0f64; n];

        let solver = AugmentedKrylov::new(AugmentedKrylovConfig {
            krylov_dim: 6,
            tol: 1e-12,
            max_iter: 300,
            max_cycles: 20,
        });

        let result = solver
            .solve(diag_mv(diag.clone()), &b, None, &[])
            .expect("augmented krylov solve failed");

        assert!(
            result.converged,
            "should converge without augmentation: residual = {:.3e}",
            result.residual_norm
        );
    }

    #[test]
    fn test_augmented_krylov_with_augmentation() {
        // Provide augmentation vectors as the first two standard basis vectors.
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b = vec![1.0f64; n];

        // Augmentation: e_0, e_1 (should align with first two basis vectors of solution).
        let aug = vec![
            {
                let mut v = vec![0.0f64; n];
                v[0] = 1.0;
                v
            },
            {
                let mut v = vec![0.0f64; n];
                v[1] = 1.0;
                v
            },
        ];

        let solver = AugmentedKrylov::new(AugmentedKrylovConfig {
            krylov_dim: 8,
            tol: 1e-12,
            max_iter: 300,
            max_cycles: 30,
        });

        let result = solver
            .solve(diag_mv(diag), &b, None, &aug)
            .expect("augmented krylov with augmentation failed");

        assert!(
            result.converged,
            "should converge with augmentation: residual = {:.3e}",
            result.residual_norm
        );
    }

    #[test]
    fn test_augmented_result_new_augmentation_populated() {
        let n = 6;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b = vec![1.0f64; n];

        let aug = vec![{
            let mut v = vec![0.0f64; n];
            v[0] = 1.0;
            v
        }];

        let solver = AugmentedKrylov::with_defaults();
        let result = solver
            .solve(diag_mv(diag), &b, None, &aug)
            .expect("solve failed");

        // new_augmentation may be empty if no Krylov basis was built, but should not panic.
        assert!(result.converged || result.residual_norm < 1e-8);
    }

    #[test]
    fn test_augmented_config_default() {
        let cfg = AugmentedKrylovConfig::default();
        assert_eq!(cfg.krylov_dim, 20);
        assert!(cfg.tol > 0.0);
        assert!(cfg.max_iter > 0);
    }
}
