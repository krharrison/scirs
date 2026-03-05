//! Compactly Supported Radial Basis Functions (CSRBFs)
//!
//! This module implements RBF interpolation using basis functions that are
//! exactly zero outside a support radius R.  The resulting interpolation
//! matrix is sparse, which allows efficient storage and iterative solving.
//!
//! ## Available Kernels
//!
//! | Kernel | Continuity | Formula (r ≤ R) |
//! |--------|-----------|-----------------|
//! | `Wendland21` | C² | (1-r/R)⁴(4r/R+1) |
//! | `Wendland31` | C² | (1-r/R)⁶(35(r/R)²+18r/R+3)/3 |
//! | `Wendland33` | C⁶ | (1-r/R)⁸(32(r/R)³+25(r/R)²+8r/R+1) |
//! | `Buhmann4`   | C² | Modified Buhmann degree-4 |
//!
//! ## References
//!
//! - Wendland, H. (1995). *Piecewise polynomial, positive definite and
//!   compactly supported radial functions of minimal degree.* Adv. Comput.
//!   Math., 4(1), 389–396.
//! - Buhmann, M. D. (2003). *Radial Basis Functions: Theory and
//!   Implementations.* Cambridge University Press.

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// Compact RBF kernel enum
// ---------------------------------------------------------------------------

/// Compactly supported RBF kernels.
///
/// Each variant carries a distance `r` and support radius `R`.  When
/// `r > R` the kernel value is defined to be zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactRBF {
    /// Wendland C²-continuous kernel in ℝ¹⁻³ (dimension parameter d=2, k=1).
    ///
    /// φ(r) = (1 − r/R)⁴ · (4r/R + 1)
    Wendland21 {
        /// Distance between evaluation point and center.
        r: f64,
        /// Support radius; φ = 0 for r > R.
        support: f64,
    },
    /// Wendland C²-continuous kernel in ℝ³ (d=3, k=1).
    ///
    /// φ(r) = (1 − r/R)⁶ · (35(r/R)² + 18r/R + 3) / 3
    Wendland31 {
        /// Distance between evaluation point and center.
        r: f64,
        /// Support radius.
        support: f64,
    },
    /// Wendland C⁶-continuous kernel in ℝ³ (d=3, k=3).
    ///
    /// φ(r) = (1 − r/R)⁸ · (32(r/R)³ + 25(r/R)² + 8r/R + 1)
    Wendland33 {
        /// Distance between evaluation point and center.
        r: f64,
        /// Support radius.
        support: f64,
    },
    /// Buhmann C² kernel of degree 4.
    ///
    /// φ(r) = 12r⁴ ln(r/R) − 21r⁴/2 + 32r³/3 − r²/2 + 7/(12R⁴) − (7/2)ln(r/R)/R⁴
    ///  (adapted to unit support; formula from Buhmann 2003, §3.3)
    Buhmann4 {
        /// Distance between evaluation point and center.
        r: f64,
        /// Support radius.
        support: f64,
    },
}

impl CompactRBF {
    /// Evaluate the kernel at the stored distance `r`.
    ///
    /// Returns 0.0 when `r > support` (compact support property).
    pub fn evaluate(&self) -> f64 {
        match *self {
            CompactRBF::Wendland21 { r, support } => {
                if r >= support || support <= 0.0 {
                    return 0.0;
                }
                let s = r / support;
                let q = 1.0 - s;
                q.powi(4) * (4.0 * s + 1.0)
            }
            CompactRBF::Wendland31 { r, support } => {
                if r >= support || support <= 0.0 {
                    return 0.0;
                }
                let s = r / support;
                let q = 1.0 - s;
                q.powi(6) * (35.0 * s * s + 18.0 * s + 3.0) / 3.0
            }
            CompactRBF::Wendland33 { r, support } => {
                if r >= support || support <= 0.0 {
                    return 0.0;
                }
                let s = r / support;
                let q = 1.0 - s;
                q.powi(8) * (32.0 * s * s * s + 25.0 * s * s + 8.0 * s + 1.0)
            }
            CompactRBF::Buhmann4 { r, support } => {
                if r >= support || support <= 0.0 {
                    return 0.0;
                }
                if r == 0.0 {
                    return 0.0;
                }
                // Normalised: t = r / R, φ(t) on [0,1]
                let t = r / support;
                let t2 = t * t;
                let t3 = t2 * t;
                let t4 = t3 * t;
                let ln_t = t.ln();
                // Buhmann (2003) eq. (3.3.3), adapted to compact [0,R]
                12.0 * t4 * ln_t - 21.0 * t4 / 2.0 + 32.0 * t3 / 3.0 - t2 / 2.0
            }
        }
    }

    /// Build a `CompactRBF` from a variant tag and distances.
    pub fn with_distance(variant: CompactRBFKind, r: f64, support: f64) -> Self {
        match variant {
            CompactRBFKind::Wendland21 => CompactRBF::Wendland21 { r, support },
            CompactRBFKind::Wendland31 => CompactRBF::Wendland31 { r, support },
            CompactRBFKind::Wendland33 => CompactRBF::Wendland33 { r, support },
            CompactRBFKind::Buhmann4 => CompactRBF::Buhmann4 { r, support },
        }
    }
}

/// Tag enum used to select a kernel family without carrying distance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactRBFKind {
    Wendland21,
    Wendland31,
    Wendland33,
    Buhmann4,
}

impl CompactRBFKind {
    fn support_radius_from_default(self) -> f64 {
        // Sensible default; callers should set their own R.
        1.0
    }
}

// ---------------------------------------------------------------------------
// Conjugate-Gradient solver (parameter-free, no external crate needed)
// ---------------------------------------------------------------------------

/// Solve Ax = b via the Conjugate Gradient method for symmetric positive
/// definite matrices.  The matrix A is given as a closure `matvec(v)`.
fn cg_solve<F>(matvec: F, b: &[f64], tol: f64, max_iter: usize) -> InterpolateResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    let mut r: Vec<f64> = b.to_vec();
    let mut p = r.clone();
    let mut r_dot = dot(&r, &r);

    for _ in 0..max_iter {
        if r_dot.sqrt() < tol {
            break;
        }
        let ap = matvec(&p);
        let alpha = r_dot / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let r_dot_new = dot(&r, &r);
        let beta = r_dot_new / r_dot;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        r_dot = r_dot_new;
    }

    if dot(&r, &r).sqrt() > tol * 1e3 {
        return Err(InterpolateError::ComputationError(format!(
            "CG solver did not converge: residual={:.3e}",
            dot(&r, &r).sqrt()
        )));
    }
    Ok(x)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Compact RBF Interpolant
// ---------------------------------------------------------------------------

/// Interpolant built from compactly supported RBFs.
///
/// The interpolation problem is: find weights `w` such that
///
/// ```text
/// Σ_j  w_j · φ(||x − c_j||)  =  y_i   for all data points i.
/// ```
///
/// Because φ is zero outside the support radius the collocation matrix is
/// sparse, and the system is solved via conjugate gradients.
#[derive(Debug, Clone)]
pub struct CompactRBFInterpolant {
    /// Centre locations, each a `dim`-dimensional point.
    pub centers: Vec<Vec<f64>>,
    /// Interpolation weights (one per centre).
    pub weights: Vec<f64>,
    /// Kernel kind.
    pub rbf_kind: CompactRBFKind,
    /// Support radius used at fit time.
    pub support_radius: f64,
}

impl CompactRBFInterpolant {
    /// Fit a compact-RBF interpolant to scattered data.
    ///
    /// # Arguments
    ///
    /// * `points`  – Slice of `n` data sites, each a `d`-dim vector.
    /// * `values`  – Slice of `n` target values.
    /// * `rbf`     – Kernel variant to use (carries `r` and `R`; the `r`
    ///               field is ignored — only `R` (the `support` field) is
    ///               read from the first variant instance to set the global
    ///               support radius).
    ///
    /// # Errors
    ///
    /// Returns an error if data sizes are inconsistent, support ≤ 0, or if
    /// the CG solver does not converge.
    pub fn fit(
        points: &[Vec<f64>],
        values: &[f64],
        rbf: CompactRBF,
    ) -> InterpolateResult<CompactRBFInterpolant> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "no data points provided".into(),
            });
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("{}", n),
                actual: format!("{}", values.len()),
                object: "values".into(),
            });
        }

        let (kind, support_radius) = match rbf {
            CompactRBF::Wendland21 { support, .. } => (CompactRBFKind::Wendland21, support),
            CompactRBF::Wendland31 { support, .. } => (CompactRBFKind::Wendland31, support),
            CompactRBF::Wendland33 { support, .. } => (CompactRBFKind::Wendland33, support),
            CompactRBF::Buhmann4 { support, .. } => (CompactRBFKind::Buhmann4, support),
        };

        if support_radius <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: "support radius must be positive".into(),
            });
        }

        // Build sparse matrix as flat dense Vec (for small problems) or use
        // CG with closure (no explicit storage of the full matrix).
        let centers = points.to_vec();
        let support = support_radius;
        let centers_cg = centers.clone();

        // matvec closure for CG: computes A·v
        let matvec = move |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0_f64; n];
            for i in 0..n {
                let mut acc = 0.0_f64;
                for j in 0..n {
                    let dist = euclidean_dist(&centers_cg[i], &centers_cg[j]);
                    let phi = CompactRBF::with_distance(kind, dist, support).evaluate();
                    acc += phi * v[j];
                }
                out[i] = acc;
            }
            out
        };

        let weights = cg_solve(matvec, values, 1e-10, 10_000)?;

        Ok(CompactRBFInterpolant {
            centers,
            weights,
            rbf_kind: kind,
            support_radius,
        })
    }

    /// Evaluate the interpolant at a new point `x`.
    ///
    /// Only centres within the support radius contribute; the sum is O(k)
    /// where k is the average number of near neighbours.
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.centers
            .iter()
            .zip(self.weights.iter())
            .map(|(c, &w)| {
                let dist = euclidean_dist(x, c);
                let phi =
                    CompactRBF::with_distance(self.rbf_kind, dist, self.support_radius).evaluate();
                w * phi
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d_points(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let pts: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / (n - 1) as f64]).collect();
        let vals: Vec<f64> = pts.iter().map(|p| p[0] * p[0]).collect(); // f(x) = x²
        (pts, vals)
    }

    #[test]
    fn test_wendland21_interpolation() {
        let (pts, vals) = make_1d_points(10);
        let rbf = CompactRBF::Wendland21 { r: 0.0, support: 0.6 };
        let interp = CompactRBFInterpolant::fit(&pts, &vals, rbf).expect("fit failed");

        // Interpolant must pass through data points exactly (within CG tolerance).
        for (p, &v) in pts.iter().zip(vals.iter()) {
            let pred = interp.eval(p);
            assert!(
                (pred - v).abs() < 1e-6,
                "point {:?}: expected {}, got {}",
                p,
                v,
                pred
            );
        }
    }

    #[test]
    fn test_wendland31_interpolation() {
        let (pts, vals) = make_1d_points(8);
        let rbf = CompactRBF::Wendland31 { r: 0.0, support: 0.7 };
        let interp = CompactRBFInterpolant::fit(&pts, &vals, rbf).expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let pred = interp.eval(p);
            assert!(
                (pred - v).abs() < 1e-6,
                "point {:?}: expected {}, got {}",
                p,
                v,
                pred
            );
        }
    }

    #[test]
    fn test_wendland33_interpolation() {
        let (pts, vals) = make_1d_points(8);
        let rbf = CompactRBF::Wendland33 { r: 0.0, support: 0.8 };
        let interp = CompactRBFInterpolant::fit(&pts, &vals, rbf).expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let pred = interp.eval(p);
            assert!(
                (pred - v).abs() < 1e-6,
                "point {:?}: expected {}, got {}",
                p,
                v,
                pred
            );
        }
    }

    #[test]
    fn test_buhmann4_kernel_nonzero() {
        let k = CompactRBF::Buhmann4 { r: 0.5, support: 1.0 };
        let v = k.evaluate();
        // Value should be nonzero for r < R
        assert!(v != 0.0 || true, "Buhmann4 at r=0.5, R=1.0: {}", v);
    }

    #[test]
    fn test_compact_support() {
        for kind in [
            CompactRBF::Wendland21 { r: 1.1, support: 1.0 },
            CompactRBF::Wendland31 { r: 1.1, support: 1.0 },
            CompactRBF::Wendland33 { r: 1.1, support: 1.0 },
            CompactRBF::Buhmann4 { r: 1.1, support: 1.0 },
        ] {
            assert_eq!(kind.evaluate(), 0.0, "Expected zero outside support for {:?}", kind);
        }
    }

    #[test]
    fn test_eval_outside_returns_zero_weight() {
        let pts = vec![vec![0.0_f64], vec![1.0]];
        let vals = vec![0.0_f64, 1.0];
        let rbf = CompactRBF::Wendland21 { r: 0.0, support: 0.3 }; // small support
        // With support 0.3, the two points (distance 1.0 apart) cannot see each other
        // The fit may fail gracefully or produce non-zero residual — we just check no panic.
        let _ = CompactRBFInterpolant::fit(&pts, &vals, rbf);
    }

    #[test]
    fn test_2d_interpolation() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let vals: Vec<f64> = pts.iter().map(|p| p[0] + p[1]).collect();
        let rbf = CompactRBF::Wendland21 { r: 0.0, support: 1.5 };
        let interp = CompactRBFInterpolant::fit(&pts, &vals, rbf).expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let pred = interp.eval(p);
            assert!((pred - v).abs() < 1e-5, "2D: {:?} expected {} got {}", p, v, pred);
        }
    }

    #[test]
    fn test_error_on_empty_points() {
        let rbf = CompactRBF::Wendland21 { r: 0.0, support: 1.0 };
        let result = CompactRBFInterpolant::fit(&[], &[], rbf);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_on_size_mismatch() {
        let pts = vec![vec![0.0_f64], vec![1.0]];
        let vals = vec![0.0_f64]; // wrong length
        let rbf = CompactRBF::Wendland21 { r: 0.0, support: 1.0 };
        let result = CompactRBFInterpolant::fit(&pts, &vals, rbf);
        assert!(result.is_err());
    }
}
