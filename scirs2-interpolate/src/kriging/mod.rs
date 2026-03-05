//! Kriging / Gaussian Process Interpolation
//!
//! This module provides **ordinary kriging** (OK) with pluggable variogram
//! models.  Kriging is a best linear unbiased estimator (BLUE) for spatial
//! data; it not only predicts values at unsampled locations but also delivers
//! an associated prediction variance.
//!
//! ## Variogram models
//!
//! | Type | Description |
//! |------|-------------|
//! | `SphericalVariogram`    | Linear near origin, plateau at `range` |
//! | `ExponentialVariogram`  | Exponential approach to sill |
//! | `GaussianVariogram`     | Smooth Gaussian approach to sill |
//! | `PowerVariogram`        | Unbounded power-law (fractal) model |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use scirs2_interpolate::kriging::{OrdinaryKriging, SphericalVariogram};
//!
//! let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
//! let values = vec![0.0, 1.0, 0.5, 0.8];
//! let vgm = SphericalVariogram { nugget: 0.0, sill: 1.0, range: 5.0 };
//! let ok = OrdinaryKriging::fit(points.clone(), values.clone(), Box::new(vgm)).expect("doc example: should succeed");
//! let (estimate, variance) = ok.predict(&[1.5]).expect("doc example: should succeed");
//! ```
//!
//! ## References
//!
//! - Cressie, N. (1993). *Statistics for Spatial Data* (revised ed.). Wiley.
//! - Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics*.
//!   Academic Press.

use crate::error::{InterpolateError, InterpolateResult};

// ---------------------------------------------------------------------------
// Variogram trait and concrete models
// ---------------------------------------------------------------------------

/// Isotropic variogram model γ(h): characterises spatial variance as a
/// function of separation distance `h ≥ 0`.
pub trait Variogram: Send + Sync {
    /// Semi-variance at lag `h`.
    fn gamma(&self, h: f64) -> f64;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Variogram>;
}

/// Spherical variogram.
///
/// γ(h) = nugget + sill · \[3h/(2r) − h³/(2r³)\]  for h ≤ r  
/// γ(h) = nugget + sill                              for h > r
#[derive(Debug, Clone, Copy)]
pub struct SphericalVariogram {
    /// Micro-scale variance (discontinuity at origin).
    pub nugget: f64,
    /// Sill: variance at large distances.
    pub sill: f64,
    /// Range: distance at which the sill is (practically) reached.
    pub range: f64,
}

impl Variogram for SphericalVariogram {
    fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        if h >= self.range {
            return self.nugget + self.sill;
        }
        let u = h / self.range;
        self.nugget + self.sill * (1.5 * u - 0.5 * u * u * u)
    }

    fn clone_box(&self) -> Box<dyn Variogram> {
        Box::new(*self)
    }
}

/// Exponential variogram.
///
/// γ(h) = nugget + sill · (1 − exp(−3h/r))
#[derive(Debug, Clone, Copy)]
pub struct ExponentialVariogram {
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
}

impl Variogram for ExponentialVariogram {
    fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        self.nugget + self.sill * (1.0 - (-3.0 * h / self.range).exp())
    }

    fn clone_box(&self) -> Box<dyn Variogram> {
        Box::new(*self)
    }
}

/// Gaussian variogram.
///
/// γ(h) = nugget + sill · (1 − exp(−3h²/r²))
#[derive(Debug, Clone, Copy)]
pub struct GaussianVariogram {
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
}

impl Variogram for GaussianVariogram {
    fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        let u = h / self.range;
        self.nugget + self.sill * (1.0 - (-3.0 * u * u).exp())
    }

    fn clone_box(&self) -> Box<dyn Variogram> {
        Box::new(*self)
    }
}

/// Power (fractal) variogram — unbounded.
///
/// γ(h) = nugget + slope · h^power  (power ∈ (0, 2))
#[derive(Debug, Clone, Copy)]
pub struct PowerVariogram {
    pub nugget: f64,
    pub slope: f64,
    pub power: f64,
}

impl Variogram for PowerVariogram {
    fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        self.nugget + self.slope * h.powf(self.power)
    }

    fn clone_box(&self) -> Box<dyn Variogram> {
        Box::new(*self)
    }
}

// ---------------------------------------------------------------------------
// LU factorisation (in-place Doolittle decomposition)
// ---------------------------------------------------------------------------

/// Doolittle LU factorisation with partial pivoting.
///
/// Returns (L·U matrix packed together, pivot indices).
fn lu_factor(mut a: Vec<f64>, n: usize) -> InterpolateResult<(Vec<f64>, Vec<usize>)> {
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Find pivot
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-15 {
            return Err(InterpolateError::ComputationError(
                "Singular kriging matrix; add nugget > 0 or check data".into(),
            ));
        }
        // Swap rows k and max_row
        if max_row != k {
            piv.swap(k, max_row);
            for j in 0..n {
                let tmp = a[k * n + j];
                a[k * n + j] = a[max_row * n + j];
                a[max_row * n + j] = tmp;
            }
        }
        // Elimination
        for i in (k + 1)..n {
            a[i * n + k] /= a[k * n + k];
            for j in (k + 1)..n {
                let tmp = a[i * n + k] * a[k * n + j];
                a[i * n + j] -= tmp;
            }
        }
    }
    Ok((a, piv))
}

/// Solve LU·x = b (in-place substitution).
fn lu_solve(lu: &[f64], piv: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    // Apply row permutation
    let mut x: Vec<f64> = (0..n).map(|i| b[piv[i]]).collect();
    // Forward substitution (L·y = b)
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[i * n + j] * x[j];
        }
    }
    // Backward substitution (U·x = y)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= lu[i * n + j] * x[j];
        }
        x[i] /= lu[i * n + i];
    }
    x
}

// ---------------------------------------------------------------------------
// Ordinary Kriging
// ---------------------------------------------------------------------------

/// Ordinary Kriging interpolant.
///
/// The kriging system of size (n+1)×(n+1) is:
///
/// ```text
/// [ C   1 ] [ w  ]   [ c₀ ]
/// [ 1ᵀ  0 ] [ μ  ] = [ 1  ]
/// ```
///
/// where `C[i,j] = cov(i,j) = (nugget+sill) - γ(||xᵢ−xⱼ||)` and
/// `c₀[i] = (nugget+sill) - γ(||x−xᵢ||)` for the prediction point `x`.
/// The nugget + sill value is the *a priori* variance (covariance at h=0⁺).
pub struct OrdinaryKriging {
    /// Data locations.
    pub points: Vec<Vec<f64>>,
    /// Observed values.
    pub values: Vec<f64>,
    /// Variogram model.
    variogram: Box<dyn Variogram>,
    /// LU factorisation of the kriging matrix (packed, row-major, size (n+1)²).
    lu_mat: Vec<f64>,
    /// Pivot vector for LU.
    lu_piv: Vec<usize>,
    /// Sill + nugget = C(0) — used when computing prediction variance.
    c0: f64,
    /// Number of data points.
    n: usize,
}

impl Clone for OrdinaryKriging {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            values: self.values.clone(),
            variogram: self.variogram.clone_box(),
            lu_mat: self.lu_mat.clone(),
            lu_piv: self.lu_piv.clone(),
            c0: self.c0,
            n: self.n,
        }
    }
}

impl std::fmt::Debug for OrdinaryKriging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryKriging")
            .field("n", &self.n)
            .finish()
    }
}

impl OrdinaryKriging {
    /// Fit the ordinary kriging interpolant to scattered data.
    ///
    /// # Arguments
    ///
    /// * `points`    – Data sites (n × d).
    /// * `values`    – Observed values at each site.
    /// * `variogram` – Variogram model implementing [`Variogram`].
    ///
    /// # Errors
    ///
    /// Returns an error if the kriging matrix is singular (add nugget > 0).
    pub fn fit(
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        variogram: Box<dyn Variogram>,
    ) -> InterpolateResult<OrdinaryKriging> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "no data points".into(),
            });
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("{}", n),
                actual: format!("{}", values.len()),
                object: "values".into(),
            });
        }

        // C(0) = γ(∞) for bounded models, or a large value for Power.
        // We estimate it as γ(very_large_distance).
        let c0 = variogram.gamma(1e12);

        // Build (n+1)×(n+1) kriging matrix
        let m = n + 1;
        let mut mat = vec![0.0_f64; m * m];
        for i in 0..n {
            for j in 0..n {
                let h = euclidean_dist(&points[i], &points[j]);
                let gamma = variogram.gamma(h);
                // Covariance = c0 - γ(h)
                mat[i * m + j] = c0 - gamma;
            }
            // Lagrange multiplier row/col
            mat[i * m + n] = 1.0;
            mat[n * m + i] = 1.0;
        }
        // Bottom-right: 0
        mat[n * m + n] = 0.0;

        let (lu_mat, lu_piv) = lu_factor(mat, m)?;

        Ok(OrdinaryKriging {
            points,
            values,
            variogram,
            lu_mat,
            lu_piv,
            c0,
            n,
        })
    }

    /// Predict at a new point `x`.
    ///
    /// Returns `(estimate, variance)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension of `x` does not match the training data.
    pub fn predict(&self, x: &[f64]) -> InterpolateResult<(f64, f64)> {
        if !self.points.is_empty() && x.len() != self.points[0].len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected dim {}, got {}",
                self.points[0].len(),
                x.len()
            )));
        }

        let m = self.n + 1;
        // Build right-hand side: [c0 - γ(||x - xᵢ||)] for i=1..n, then 1
        let mut rhs = vec![0.0_f64; m];
        for i in 0..self.n {
            let h = euclidean_dist(x, &self.points[i]);
            rhs[i] = self.c0 - self.variogram.gamma(h);
        }
        rhs[self.n] = 1.0;

        // Solve kriging system
        let sol = lu_solve(&self.lu_mat, &self.lu_piv, &rhs, m);

        // Estimate: Σ wᵢ · zᵢ
        let estimate: f64 = (0..self.n).map(|i| sol[i] * self.values[i]).sum();

        // Kriging variance: σ² = c0 - Σ wᵢ·(c0 - γᵢ) - μ
        //   = c0 - c^T w - μ   (μ = sol[n])
        let ct_w: f64 = (0..self.n).map(|i| rhs[i] * sol[i]).sum();
        let variance = (self.c0 - ct_w - sol[self.n]).max(0.0);

        Ok((estimate, variance))
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

    fn make_1d_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let pts: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        let vals: Vec<f64> = xs.iter().map(|&x| x * x).collect(); // f(x)=x²
        (pts, vals)
    }

    #[test]
    fn test_spherical_kriging_interpolates_data() {
        let (pts, vals) = make_1d_data();
        let vgm = SphericalVariogram { nugget: 0.0, sill: 20.0, range: 10.0 };
        let ok = OrdinaryKriging::fit(pts.clone(), vals.clone(), Box::new(vgm))
            .expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (est, _var) = ok.predict(p).expect("predict failed");
            assert!(
                (est - v).abs() < 1e-6,
                "spherical: at {:?} expected {} got {}",
                p,
                v,
                est
            );
        }
    }

    #[test]
    fn test_exponential_kriging_interpolates_data() {
        let (pts, vals) = make_1d_data();
        let vgm = ExponentialVariogram { nugget: 0.0, sill: 20.0, range: 10.0 };
        let ok = OrdinaryKriging::fit(pts.clone(), vals.clone(), Box::new(vgm))
            .expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (est, _) = ok.predict(p).expect("predict");
            assert!((est - v).abs() < 1e-6, "exp: {:?} {} {}", p, v, est);
        }
    }

    #[test]
    fn test_gaussian_kriging_interpolates_data() {
        let (pts, vals) = make_1d_data();
        let vgm = GaussianVariogram { nugget: 0.0, sill: 20.0, range: 10.0 };
        let ok = OrdinaryKriging::fit(pts.clone(), vals.clone(), Box::new(vgm))
            .expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (est, _) = ok.predict(p).expect("predict");
            assert!((est - v).abs() < 1e-6, "gauss: {:?} {} {}", p, v, est);
        }
    }

    #[test]
    fn test_power_variogram() {
        let (pts, vals) = make_1d_data();
        let vgm = PowerVariogram { nugget: 0.0, slope: 1.0, power: 1.5 };
        let ok = OrdinaryKriging::fit(pts.clone(), vals.clone(), Box::new(vgm))
            .expect("fit failed");

        for (p, &v) in pts.iter().zip(vals.iter()) {
            let (est, _) = ok.predict(p).expect("predict");
            assert!((est - v).abs() < 1e-4, "power: {:?} {} {}", p, v, est);
        }
    }

    #[test]
    fn test_variance_is_nonnegative() {
        let (pts, vals) = make_1d_data();
        let vgm = SphericalVariogram { nugget: 0.01, sill: 20.0, range: 10.0 };
        let ok = OrdinaryKriging::fit(pts, vals, Box::new(vgm)).expect("fit failed");
        let test_pts = vec![vec![0.5_f64], vec![1.5], vec![2.5]];
        for p in &test_pts {
            let (_est, var) = ok.predict(p).expect("predict");
            assert!(var >= 0.0, "variance negative at {:?}: {}", p, var);
        }
    }

    #[test]
    fn test_variogram_gamma_at_zero() {
        let svgm = SphericalVariogram { nugget: 0.1, sill: 1.0, range: 2.0 };
        assert_eq!(svgm.gamma(0.0), 0.0);
        let evgm = ExponentialVariogram { nugget: 0.1, sill: 1.0, range: 2.0 };
        assert_eq!(evgm.gamma(0.0), 0.0);
        let gvgm = GaussianVariogram { nugget: 0.1, sill: 1.0, range: 2.0 };
        assert_eq!(gvgm.gamma(0.0), 0.0);
        let pvgm = PowerVariogram { nugget: 0.1, slope: 1.0, power: 1.5 };
        assert_eq!(pvgm.gamma(0.0), 0.0);
    }

    #[test]
    fn test_spherical_reaches_sill() {
        let vgm = SphericalVariogram { nugget: 0.0, sill: 5.0, range: 2.0 };
        let v = vgm.gamma(100.0);
        assert!((v - 5.0).abs() < 1e-10, "should reach sill: {}", v);
    }

    #[test]
    fn test_error_on_empty() {
        let vgm = SphericalVariogram { nugget: 0.0, sill: 1.0, range: 1.0 };
        let r = OrdinaryKriging::fit(vec![], vec![], Box::new(vgm));
        assert!(r.is_err());
    }

    #[test]
    fn test_error_on_dim_mismatch_predict() {
        let pts = vec![vec![0.0_f64, 0.0], vec![1.0, 1.0]];
        let vals = vec![0.0_f64, 1.0];
        let vgm = GaussianVariogram { nugget: 0.0, sill: 1.0, range: 5.0 };
        let ok = OrdinaryKriging::fit(pts, vals, Box::new(vgm)).expect("fit");
        let r = ok.predict(&[0.5]); // wrong dim
        assert!(r.is_err());
    }
}
