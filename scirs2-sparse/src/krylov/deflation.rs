//! Deflation techniques for Krylov methods.
//!
//! Deflation removes the influence of a known (approximately) invariant subspace
//! from the operator before or during the Krylov iteration, enabling the method
//! to focus on the remaining (harder-to-converge) part of the spectrum.
//!
//! # Deflation strategies
//!
//! ## Harmonic Ritz Deflation
//!
//! Extracts harmonic Ritz pairs (θ, y) from a Krylov basis V_m such that
//!
//!   A V_m y ≈ θ V_m y,
//!
//! and uses the corresponding Ritz vectors to build a deflation projector.
//! Deflating small harmonic Ritz values dramatically reduces the iteration
//! count for systems whose convergence stagnation is caused by small eigenvalues.
//!
//! ## Projective Deflation (Nicolaides / Frank 1977 / Vuik 1999)
//!
//! Given an approximate invariant subspace Z such that A Z ≈ Z Λ, the
//! deflated operator is
//!
//!   A_d = A (I - Z Z^+)  +  Z Z^+
//!
//! which shifts the small eigenvalues to 1 without affecting the rest.
//!
//! # References
//!
//! - Morgan, R.B. (1995). "A restarted GMRES method augmented with eigenvectors".
//!   SIAM J. Matrix Anal. Appl. 16(4), 1154-1171.
//! - de Sturler, E. (1996). "A performance model for Krylov subspace methods on
//!   parallel computers". Parallel Comput. 22(10), 1337-1352.

use crate::error::SparseError;
use crate::krylov::gmres_dr::{dot, gram_schmidt_mgs, norm2};

/// Harmonic Ritz deflation operator.
///
/// Maintains a set of approximate eigenvectors corresponding to the smallest
/// (in absolute value) harmonic Ritz values extracted from a Krylov basis.
/// These vectors can be:
///
/// 1. Used as starting augmentation vectors in the next GMRES restart (GMRES-DR).
/// 2. Applied as a projective deflation preconditioner.
/// 3. Fed into [`AugmentedKrylov`] as initial search directions.
///
/// [`AugmentedKrylov`]: super::augmented::AugmentedKrylov
#[derive(Debug, Clone)]
pub struct HarmonicRitzDeflation {
    /// Maximum number of deflation vectors to maintain.
    pub n_deflate: usize,
    /// Deflation (Ritz) vectors. Each vector is normalised; together they are
    /// orthonormal.
    pub vectors: Vec<Vec<f64>>,
    /// Approximate eigenvalues corresponding to each deflation vector.
    pub values: Vec<f64>,
}

impl HarmonicRitzDeflation {
    /// Create a new (empty) deflation object targeting `n_deflate` vectors.
    pub fn new(n_deflate: usize) -> Self {
        Self {
            n_deflate,
            vectors: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Extract harmonic Ritz pairs from a converged Krylov subspace.
    ///
    /// # Arguments
    ///
    /// * `h` - Upper-Hessenberg matrix of size (m+1) × m.
    /// * `v` - Arnoldi basis vectors `v[0..m+1]`, each of length n.
    /// * `m` - Actual Krylov dimension (number of Arnoldi steps taken).
    ///
    /// # Algorithm
    ///
    /// The harmonic Ritz values are the eigenvalues of the m×m matrix
    ///
    ///   B_m = H_m + h_{m+1,m}^2 * H_m^{-T} e_m e_m^T
    ///
    /// We approximate this via the QR algorithm applied to H_m.  The `n_deflate`
    /// Ritz vectors corresponding to the smallest |eigenvalue| are retained.
    pub fn extract_from_krylov(&mut self, h: &[Vec<f64>], v: &[Vec<f64>], m: usize) {
        if m == 0 {
            return;
        }
        let n = if !v.is_empty() { v[0].len() } else { return };
        let n_take = self.n_deflate.min(m);

        // Build H_m (m×m square submatrix) and the harmonic perturbation.
        let h_extra = if m < h.len() && m.checked_sub(1).is_some() {
            let col = m - 1;
            if col < h[m].len() {
                h[m][col].abs()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let hm: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                if i < h.len() {
                    let row_len = m.min(h[i].len());
                    let mut row = h[i][..row_len].to_vec();
                    row.resize(m, 0.0);
                    row
                } else {
                    vec![0.0; m]
                }
            })
            .collect();

        // Compute harmonic Ritz vectors: eigenvectors of H_m corresponding to
        // eigenvalues with smallest |value| (these deflate the slow modes).
        let schur_vecs = harmonic_ritz_schur_vecs(&hm, h_extra, n_take);

        // Map back: y = V_m s (linear combination of basis vectors).
        let mut new_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_take);
        let mut new_vals: Vec<f64> = Vec::with_capacity(n_take);

        for (eig_val, s) in &schur_vecs {
            if s.len() != m {
                continue;
            }
            let mut y = vec![0.0f64; n];
            for (j, &sj) in s.iter().enumerate() {
                if j < v.len() {
                    for l in 0..n {
                        y[l] += sj * v[j][l];
                    }
                }
            }
            let nrm = norm2(&y);
            if nrm > 1e-15 {
                for yi in &mut y {
                    *yi /= nrm;
                }
                new_vecs.push(y);
                new_vals.push(*eig_val);
            }
        }

        // Orthonormalise to ensure numerical stability.
        gram_schmidt_mgs(&mut new_vecs);

        self.vectors = new_vecs;
        self.values = new_vals;
    }

    /// Apply projective deflation: remove the component of `r` in the deflation subspace.
    ///
    /// Returns r_d = r - V (V^T r) where V is the deflation basis.
    ///
    /// This can be used as a post-step correction or as part of a deflation preconditioner.
    pub fn deflate(&self, r: &[f64]) -> Vec<f64> {
        let mut result = r.to_vec();
        for v in &self.vectors {
            let coeff = dot(&result, v);
            let norm2_v: f64 = dot(v, v);
            if norm2_v > 1e-300 {
                let scale = coeff / norm2_v;
                for (ri, vi) in result.iter_mut().zip(v.iter()) {
                    *ri -= scale * vi;
                }
            }
        }
        result
    }

    /// Apply deflation-based correction to a solution vector.
    ///
    /// Given a system A x = b and an approximate invariant subspace V such that
    /// A V ≈ V Λ, compute the correction:
    ///
    ///   Δx = V Λ^{-1} V^T r
    ///
    /// where r = b - A x is the current residual.
    ///
    /// # Arguments
    ///
    /// * `r` - Current residual vector.
    /// * `av` - A * v for each deflation vector v. Must have same length as `self.vectors`.
    ///
    /// # Returns
    ///
    /// The correction Δx to add to the current iterate.
    pub fn correction_from_residual(&self, r: &[f64], av: &[Vec<f64>]) -> Vec<f64> {
        let n = r.len();
        let k = self.vectors.len().min(av.len());
        let mut delta = vec![0.0f64; n];

        for j in 0..k {
            let av_j = &av[j];
            let av_norm2 = dot(av_j, av_j);
            if av_norm2 < 1e-300 {
                continue;
            }
            let coeff = dot(av_j, r) / av_norm2;
            for i in 0..n {
                delta[i] += coeff * self.vectors[j][i];
            }
        }

        delta
    }

    /// Return the number of deflation vectors currently stored.
    pub fn dim(&self) -> usize {
        self.vectors.len()
    }

    /// Remove all deflation vectors.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.values.clear();
    }

    /// Check whether the deflation vectors adequately span the given residual.
    ///
    /// Returns the relative projection norm: ||P_V r|| / ||r|| where P_V is the
    /// projector onto span(V).  A value close to 1 means r is mostly in the
    /// deflation subspace.
    pub fn projection_quality(&self, r: &[f64]) -> f64 {
        let r_norm = norm2(r);
        if r_norm < 1e-300 {
            return 0.0;
        }
        let mut proj = 0.0f64;
        for v in &self.vectors {
            let c = dot(r, v);
            let vn2 = dot(v, v);
            if vn2 > 1e-300 {
                proj += c * c / vn2;
            }
        }
        proj.sqrt() / r_norm
    }
}

/// Compute harmonic Ritz vectors from the m×m Hessenberg matrix.
///
/// Returns at most `n_take` pairs (eigenvalue, eigenvector) corresponding to
/// the eigenvalues of smallest absolute value.
///
/// Uses a power-deflation method: iterative inverse-iteration on H_m to find
/// eigenvectors.
fn harmonic_ritz_schur_vecs(hm: &[Vec<f64>], h_extra: f64, n_take: usize) -> Vec<(f64, Vec<f64>)> {
    let m = hm.len();
    if m == 0 || n_take == 0 {
        return Vec::new();
    }

    // Build the harmonic modification: B = H_m + h_extra^2 * H_m^{-T} e_m e_m^T.
    // Rather than computing H_m^{-T} explicitly, we estimate the harmonic Ritz values
    // from the eigenvalues of H_m itself (for the deflation application the exact
    // harmonic Ritz values are not required — we only need good approximate directions).
    //
    // Algorithm: apply shifted inverse power iteration to H_m to find the
    // eigenvectors corresponding to the n_take eigenvalues closest to zero.

    // First, estimate eigenvalues via the Schur form of H_m.
    let mut a = hm.to_vec();

    // Incorporate harmonic shift in bottom-right corner.
    if h_extra > 1e-15 && m >= 1 {
        a[m - 1][m - 1] += h_extra * h_extra;
    }

    // Run QR iteration with shifts to get approximate eigenvalues from diagonal.
    let n_qr = 30 * m;
    let mut q_total: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = vec![0.0f64; m];
            row[i] = 1.0;
            row
        })
        .collect();

    for _ in 0..n_qr {
        let shift = if m >= 2 { a[m - 1][m - 1] } else { 0.0 };
        let (q, r) = hessenberg_qr(&a, shift, m);
        a = dense_mat_mul(&r, &q, m);
        for i in 0..m {
            a[i][i] += shift;
        }
        q_total = dense_mat_mul(&q_total, &q, m);
    }

    // Collect eigenvalues (diagonal of a after QR convergence) and sort by |value|.
    let mut eig_pairs: Vec<(f64, usize)> = (0..m).map(|i| (a[i][i].abs(), i)).collect();
    eig_pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let take = n_take.min(m);
    eig_pairs[..take]
        .iter()
        .map(|(_, col)| {
            let eig = a[*col][*col];
            let s: Vec<f64> = (0..m).map(|row| q_total[row][*col]).collect();
            (eig, s)
        })
        .collect()
}

/// Perform one QR step on the m×m (nearly upper-triangular) matrix `a` with
/// Wilkinson shift.  Returns (Q, R) such that a - shift*I = Q R.
fn hessenberg_qr(a: &[Vec<f64>], shift: f64, m: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut r: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = a[i].clone();
            row.resize(m, 0.0);
            if i < row.len() {
                row[i] -= shift;
            }
            row
        })
        .collect();

    let mut q: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = vec![0.0f64; m];
            row[i] = 1.0;
            row
        })
        .collect();

    for j in 0..m.saturating_sub(1) {
        let a_jj = if j < r.len() && j < r[j].len() {
            r[j][j]
        } else {
            0.0
        };
        let a_j1j = if j + 1 < r.len() && j < r[j + 1].len() {
            r[j + 1][j]
        } else {
            0.0
        };
        let denom = (a_jj * a_jj + a_j1j * a_j1j).sqrt();
        let (c, s) = if denom < 1e-300 {
            (1.0f64, 0.0f64)
        } else {
            (a_jj / denom, a_j1j / denom)
        };

        for col in 0..m {
            let r_jc = if j < r.len() && col < r[j].len() {
                r[j][col]
            } else {
                0.0
            };
            let r_j1c = if j + 1 < r.len() && col < r[j + 1].len() {
                r[j + 1][col]
            } else {
                0.0
            };
            if j < r.len() && col < r[j].len() {
                r[j][col] = c * r_jc + s * r_j1c;
            }
            if j + 1 < r.len() && col < r[j + 1].len() {
                r[j + 1][col] = -s * r_jc + c * r_j1c;
            }
        }
        for row in 0..m {
            let q_rj = if row < q.len() && j < q[row].len() {
                q[row][j]
            } else {
                0.0
            };
            let q_rj1 = if row < q.len() && j + 1 < q[row].len() {
                q[row][j + 1]
            } else {
                0.0
            };
            if row < q.len() && j < q[row].len() {
                q[row][j] = c * q_rj + s * q_rj1;
            }
            if row < q.len() && j + 1 < q[row].len() {
                q[row][j + 1] = -s * q_rj + c * q_rj1;
            }
        }
    }

    (q, r)
}

/// Dense m×m matrix multiply.
fn dense_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], m: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for k in 0..m {
            let a_ik = if i < a.len() && k < a[i].len() {
                a[i][k]
            } else {
                0.0
            };
            if a_ik.abs() < 1e-300 {
                continue;
            }
            for j in 0..m {
                let b_kj = if k < b.len() && j < b[k].len() {
                    b[k][j]
                } else {
                    0.0
                };
                c[i][j] += a_ik * b_kj;
            }
        }
    }
    c
}

/// Check that the solve using deflation actually reduces residual.
pub fn deflation_reduces_residual<F>(
    matvec: F,
    deflation: &HarmonicRitzDeflation,
    b: &[f64],
    x: &[f64],
) -> bool
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let ax = matvec(x);
    let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
    let r_deflated = deflation.deflate(&r);
    norm2(&r_deflated) <= norm2(&r) + 1e-12
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_ritz_deflation_construction() {
        let defl = HarmonicRitzDeflation::new(4);
        assert_eq!(defl.n_deflate, 4);
        assert!(defl.vectors.is_empty());
        assert!(defl.values.is_empty());
        assert_eq!(defl.dim(), 0);
    }

    #[test]
    fn test_deflate_zero_vector() {
        let defl = HarmonicRitzDeflation::new(2);
        let r = vec![0.0f64; 5];
        let deflated = defl.deflate(&r);
        assert_eq!(deflated, r);
    }

    #[test]
    fn test_deflate_reduces_norm() {
        // With one deflation vector e_0 = [1,0,0,...], deflating r = [1,1,1,...] should remove
        // the first component: result = [0, 1, 1, ...].
        let mut defl = HarmonicRitzDeflation::new(1);
        defl.vectors = vec![vec![1.0, 0.0, 0.0, 0.0]];
        defl.values = vec![0.1];

        let r = vec![1.0, 1.0, 1.0, 1.0];
        let r_d = defl.deflate(&r);

        // Component along e_0 should be removed.
        assert!(r_d[0].abs() < 1e-14, "r_d[0] = {}", r_d[0]);
        assert!((r_d[1] - 1.0).abs() < 1e-14);
        assert!((r_d[2] - 1.0).abs() < 1e-14);
        assert!((r_d[3] - 1.0).abs() < 1e-14);

        // Norm reduced.
        assert!(norm2(&r_d) < norm2(&r));
    }

    #[test]
    fn test_extract_from_krylov_trivial() {
        // With a trivial 2×2 Hessenberg, should extract 1 deflation vector.
        let mut defl = HarmonicRitzDeflation::new(1);
        let h = vec![vec![2.0, 1.0], vec![0.5, 1.5], vec![0.1, 0.0]];
        let v = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        defl.extract_from_krylov(&h, &v, 2);

        // Should have extracted at most 1 vector.
        assert!(defl.vectors.len() <= 1);
        // Extracted vector should be normalised.
        for vi in &defl.vectors {
            let nrm = norm2(vi);
            assert!(
                (nrm - 1.0).abs() < 1e-12,
                "deflation vector not normalised: {}",
                nrm
            );
        }
    }

    #[test]
    fn test_deflation_projection_quality_orthogonal() {
        // If r is exactly in span(V), projection quality should be ~1.
        let mut defl = HarmonicRitzDeflation::new(2);
        defl.vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        defl.values = vec![0.1, 0.2];

        let r = vec![3.0, 4.0, 0.0];
        let q = defl.projection_quality(&r);
        let expected = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt() / norm2(&r);
        assert!(
            (q - expected).abs() < 1e-12,
            "q = {:.6}, expected = {:.6}",
            q,
            expected
        );
    }

    #[test]
    fn test_deflation_clear() {
        let mut defl = HarmonicRitzDeflation::new(3);
        defl.vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        defl.values = vec![0.1, 0.2];
        defl.clear();
        assert_eq!(defl.dim(), 0);
        assert!(defl.values.is_empty());
    }

    #[test]
    fn test_correction_from_residual() {
        // With one deflation vector u and Au, correction should reduce residual component.
        let n = 4;
        let u = vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0, 0.0];
        let lambda = 0.5f64;
        let au: Vec<f64> = u.iter().map(|x| lambda * x).collect();

        let mut defl = HarmonicRitzDeflation::new(1);
        defl.vectors = vec![u.clone()];
        defl.values = vec![lambda];
        let av_list = vec![au.clone()];

        // Residual in direction of u.
        let r = vec![2.0, 2.0, 0.0, 0.0];
        let correction = defl.correction_from_residual(&r, &av_list);

        assert_eq!(correction.len(), n);
        // Correction should point along u.
        let proj_c_on_u = dot(&correction, &u);
        assert!(
            proj_c_on_u.abs() > 1e-10,
            "correction should have non-zero projection on u"
        );
    }
}
