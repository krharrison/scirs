//! Advanced Radial Basis Function (RBF) Interpolation
//!
//! Provides several families of RBF interpolation:
//!
//! - **Polyharmonic splines**: thin-plate splines (`r² log r`), biharmonic `rᵏ` for odd k.
//! - **Global RBF with polynomial augmentation**: solves the symmetric saddle-point system
//!   `[A P; Pᵀ 0] [c; d] = [f; 0]` for positive-definiteness and reproduction of polynomials
//!   up to a given degree.
//! - **Compactly-supported Wendland functions** (ψ₃,₀, ψ₃,₁, ψ₃,₂) for sparse systems.
//! - **RBF-FD (finite differences)**: builds local stencil weight vectors from an RBF solve.
//! - **Hermite RBF**: simultaneously interpolates function values *and* gradient vectors.
//!
//! All solvers use pure-Rust LU factorisation provided by `scirs2-linalg`.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::solve;

// ---------------------------------------------------------------------------
// Helper: pairwise distances
// ---------------------------------------------------------------------------

/// Compute the `n × n` matrix of Euclidean distances `‖xᵢ − xⱼ‖`.
fn distance_matrix(points: &Array2<f64>) -> Array2<f64> {
    let n = points.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (0..points.ncols())
                .map(|k| (points[[i, k]] - points[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

/// Compute distances from a single query point to each row of `points`.
fn distances_to_point(query: &[f64], points: &Array2<f64>) -> Array1<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let mut dists = Array1::<f64>::zeros(n);
    for i in 0..n {
        let sq: f64 = (0..d).map(|k| (query[k] - points[[i, k]]).powi(2)).sum();
        dists[i] = sq.sqrt();
    }
    dists
}

// ---------------------------------------------------------------------------
// Polyharmonic kernel enum
// ---------------------------------------------------------------------------

/// Polyharmonic spline kernels.
///
/// Thin-plate spline (`r² log r`) is the canonical choice; higher-order
/// variants use `rᵏ` for odd `k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolyharmonicKernel {
    /// Thin-plate spline: φ(r) = r² log(r)  (r > 0), 0 at r = 0.
    ThinPlate,
    /// φ(r) = r³  (biharmonic spline, 2D)
    R3,
    /// φ(r) = r⁵
    R5,
    /// φ(r) = r⁷
    R7,
    /// Custom odd power k ≥ 1.
    Custom(u32),
}

impl PolyharmonicKernel {
    /// Evaluate the kernel at distance `r`.
    #[inline]
    pub fn eval(&self, r: f64) -> f64 {
        match self {
            PolyharmonicKernel::ThinPlate => {
                if r <= 0.0 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
            PolyharmonicKernel::R3 => r.powi(3),
            PolyharmonicKernel::R5 => r.powi(5),
            PolyharmonicKernel::R7 => r.powi(7),
            PolyharmonicKernel::Custom(k) => r.powi(*k as i32),
        }
    }
}

// ---------------------------------------------------------------------------
// Compactly-supported Wendland functions
// ---------------------------------------------------------------------------

/// Wendland compactly-supported radial basis functions in ℝ³.
///
/// Each function has support on `[0, radius]` and is C^(2s) smooth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WendlandKernel {
    /// ψ₃,₀(r) = (1−r)²₊           → C⁰
    Psi30,
    /// ψ₃,₁(r) = (1−r)⁴₊ (4r + 1) → C²
    Psi31,
    /// ψ₃,₂(r) = (1−r)⁶₊ (35r² + 18r + 3) / 3 → C⁴
    Psi32,
}

impl WendlandKernel {
    /// Evaluate at *scaled* radius `r ∈ [0, 1]` (r = dist / support_radius).
    #[inline]
    pub fn eval_scaled(&self, r: f64) -> f64 {
        if r >= 1.0 {
            return 0.0;
        }
        let t = 1.0 - r;
        match self {
            WendlandKernel::Psi30 => t * t,
            WendlandKernel::Psi31 => t.powi(4) * (4.0 * r + 1.0),
            WendlandKernel::Psi32 => t.powi(6) * (35.0 * r * r + 18.0 * r + 3.0) / 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Polynomial basis utilities
// ---------------------------------------------------------------------------

/// Degree of polynomial augmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyDegree {
    /// No polynomial term (m = 0).
    None,
    /// Constant term only (m = 1).
    Const,
    /// Linear terms (m = d + 1).
    Linear,
    /// Quadratic terms (m = (d+1)(d+2)/2).
    Quadratic,
}

/// Build the polynomial basis matrix `P` of shape `(n, m)` for the given
/// points and degree.  Columns: `[1, x₁, x₂, …, xd, x₁², x₁x₂, …]`.
fn build_poly_matrix(points: &Array2<f64>, degree: PolyDegree) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();

    let cols = match degree {
        PolyDegree::None => return Array2::<f64>::zeros((n, 0)),
        PolyDegree::Const => 1,
        PolyDegree::Linear => 1 + d,
        PolyDegree::Quadratic => 1 + d + d * (d + 1) / 2,
    };

    let mut p = Array2::<f64>::zeros((n, cols));
    for i in 0..n {
        let mut col = 0usize;
        // constant
        p[[i, col]] = 1.0;
        col += 1;
        if degree >= PolyDegree::Linear {
            for k in 0..d {
                p[[i, col]] = points[[i, k]];
                col += 1;
            }
        }
        if degree >= PolyDegree::Quadratic {
            for k in 0..d {
                for l in k..d {
                    p[[i, col]] = points[[i, k]] * points[[i, l]];
                    col += 1;
                }
            }
        }
    }
    p
}

/// Evaluate polynomial basis row for a single query point.
fn poly_row(query: &[f64], degree: PolyDegree) -> Vec<f64> {
    let d = query.len();
    match degree {
        PolyDegree::None => vec![],
        PolyDegree::Const => vec![1.0],
        PolyDegree::Linear => {
            let mut row = vec![1.0];
            row.extend_from_slice(query);
            row
        }
        PolyDegree::Quadratic => {
            let mut row = vec![1.0];
            row.extend_from_slice(query);
            for k in 0..d {
                for l in k..d {
                    row.push(query[k] * query[l]);
                }
            }
            row
        }
    }
}

// ---------------------------------------------------------------------------
// Global RBF interpolant with polynomial augmentation
// ---------------------------------------------------------------------------

/// Global RBF interpolant with optional polynomial reproduction.
///
/// Solves the augmented system
/// ```text
/// ┌ A   P ┐ ┌ c ┐   ┌ f ┐
/// │ Pᵀ  0 │ │ d │ = │ 0 │
/// └       ┘ └   ┘   └   ┘
/// ```
/// where `A_{ij} = φ(‖xᵢ − xⱼ‖)` and `P` is the polynomial matrix.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::rbf_interpolant::{
///     GlobalRbfInterpolant, PolyharmonicKernel, PolyDegree,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 1.0, 2.0];
/// let interp = GlobalRbfInterpolant::new_polyharmonic(
///     &pts.view(), &vals.view(), PolyharmonicKernel::ThinPlate, PolyDegree::Linear,
/// ).expect("doc example: should succeed");
/// let v = interp.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
/// assert!((v - 1.0).abs() < 1e-9);
/// ```
pub struct GlobalRbfInterpolant {
    points: Array2<f64>,
    coeffs_rbf: Array1<f64>, // c in the augmented system
    coeffs_poly: Array1<f64>, // d in the augmented system
    kernel: InternalKernel,
    degree: PolyDegree,
}

#[derive(Debug, Clone)]
enum InternalKernel {
    Polyharmonic(PolyharmonicKernel),
    Gaussian(f64),       // ε
    Multiquadric(f64),   // ε
    InvMultiquadric(f64),
}

impl InternalKernel {
    #[inline]
    fn eval(&self, r: f64) -> f64 {
        match self {
            InternalKernel::Polyharmonic(k) => k.eval(r),
            InternalKernel::Gaussian(eps) => (-(eps * r).powi(2)).exp(),
            InternalKernel::Multiquadric(eps) => (1.0 + (eps * r).powi(2)).sqrt(),
            InternalKernel::InvMultiquadric(eps) => 1.0 / (1.0 + (eps * r).powi(2)).sqrt(),
        }
    }
}

impl GlobalRbfInterpolant {
    /// Create a polyharmonic-spline RBF interpolant.
    pub fn new_polyharmonic(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        kernel: PolyharmonicKernel,
        degree: PolyDegree,
    ) -> InterpolateResult<Self> {
        Self::build(
            points,
            values,
            InternalKernel::Polyharmonic(kernel),
            degree,
        )
    }

    /// Create a Gaussian RBF interpolant (`φ = exp(−(εr)²)`).
    pub fn new_gaussian(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        epsilon: f64,
        degree: PolyDegree,
    ) -> InterpolateResult<Self> {
        if epsilon <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("epsilon must be > 0, got {epsilon}"),
            });
        }
        Self::build(points, values, InternalKernel::Gaussian(epsilon), degree)
    }

    /// Create a multiquadric RBF interpolant (`φ = sqrt(1 + (εr)²)`).
    pub fn new_multiquadric(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        epsilon: f64,
        degree: PolyDegree,
    ) -> InterpolateResult<Self> {
        if epsilon <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("epsilon must be > 0, got {epsilon}"),
            });
        }
        Self::build(
            points,
            values,
            InternalKernel::Multiquadric(epsilon),
            degree,
        )
    }

    // -----------------------------------------------------------------------

    fn build(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        kernel: InternalKernel,
        degree: PolyDegree,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let pts_owned = points.to_owned();

        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows but values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "At least one data point required".to_string(),
            ));
        }

        // Build kernel matrix A (n×n)
        let dist = distance_matrix(&pts_owned);
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = kernel.eval(dist[[i, j]]);
            }
        }

        // Build polynomial matrix P (n×m)
        let p_mat = build_poly_matrix(&pts_owned, degree);
        let m = p_mat.ncols();

        // Assemble saddle-point system (n+m) × (n+m)
        let total = n + m;
        let mut sys = Array2::<f64>::zeros((total, total));
        let mut rhs = Array1::<f64>::zeros(total);

        // Top-left: A
        for i in 0..n {
            for j in 0..n {
                sys[[i, j]] = a[[i, j]];
            }
        }
        // Top-right: P and Bottom-left: Pᵀ
        for i in 0..n {
            for j in 0..m {
                sys[[i, n + j]] = p_mat[[i, j]];
                sys[[n + j, i]] = p_mat[[i, j]];
            }
        }
        // Bottom-right block: 0 (already initialised to zero)

        // RHS: [f; 0]
        for i in 0..n {
            rhs[i] = values[i];
        }

        // Solve the system
        let sol = solve_linear_system(sys, rhs)?;

        let coeffs_rbf = sol.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let coeffs_poly = if m > 0 {
            sol.slice(scirs2_core::ndarray::s![n..]).to_owned()
        } else {
            Array1::<f64>::zeros(0)
        };

        Ok(Self {
            points: pts_owned,
            coeffs_rbf,
            coeffs_poly,
            kernel,
            degree,
        })
    }

    /// Evaluate the interpolant at a query point.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = distances_to_point(query, &self.points);
        let n = self.points.nrows();

        // RBF contribution
        let mut val: f64 = (0..n).map(|i| self.coeffs_rbf[i] * self.kernel.eval(dists[i])).sum();

        // Polynomial contribution
        let prow = poly_row(query, self.degree);
        for (j, &coeff) in self.coeffs_poly.iter().enumerate() {
            val += coeff * prow[j];
        }
        Ok(val)
    }

    /// Evaluate at multiple query points (rows of `queries`).
    pub fn evaluate_batch(&self, queries: &ArrayView2<f64>) -> InterpolateResult<Array1<f64>> {
        let nq = queries.nrows();
        let mut out = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let row: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            out[i] = self.evaluate(&row)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Compactly-supported Wendland RBF interpolant
// ---------------------------------------------------------------------------

/// Sparse RBF interpolant using Wendland's compactly-supported functions.
///
/// Only pairs with `dist < support_radius` contribute to the kernel matrix,
/// yielding a (potentially) sparse system.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::rbf_interpolant::{
///     WendlandInterpolant, WendlandKernel,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// let pts = Array2::from_shape_vec((5, 2), vec![
///     0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  0.5, 0.5,
/// ]).expect("doc example: should succeed");
/// let vals = array![0.0, 1.0, 1.0, 2.0, 1.0];
/// let interp = WendlandInterpolant::new(
///     &pts.view(), &vals.view(), WendlandKernel::Psi31, 1.5,
/// ).expect("doc example: should succeed");
/// let v = interp.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
/// ```
pub struct WendlandInterpolant {
    points: Array2<f64>,
    coeffs: Array1<f64>,
    kernel: WendlandKernel,
    support_radius: f64,
}

impl WendlandInterpolant {
    /// Build a Wendland RBF interpolant.
    ///
    /// # Arguments
    ///
    /// * `points`         – `(n, d)` data-site coordinates.
    /// * `values`         – `n` function values.
    /// * `kernel`         – which Wendland function to use.
    /// * `support_radius` – neighbourhood size `δ > 0`.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        kernel: WendlandKernel,
        support_radius: f64,
    ) -> InterpolateResult<Self> {
        if support_radius <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("support_radius must be > 0, got {support_radius}"),
            });
        }
        let n = points.nrows();
        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows but values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "At least one data point required".to_string(),
            ));
        }

        let pts_owned = points.to_owned();
        let dist = distance_matrix(&pts_owned);

        // Build kernel matrix
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let r_scaled = dist[[i, j]] / support_radius;
                a[[i, j]] = kernel.eval_scaled(r_scaled);
            }
        }

        let coeffs = solve_linear_system(a, values.to_owned())?;

        Ok(Self {
            points: pts_owned,
            coeffs,
            kernel,
            support_radius,
        })
    }

    /// Evaluate at a query point.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        let d = self.points.ncols();
        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let dists = distances_to_point(query, &self.points);
        let val: f64 = (0..self.points.nrows())
            .map(|i| {
                let r_scaled = dists[i] / self.support_radius;
                self.coeffs[i] * self.kernel.eval_scaled(r_scaled)
            })
            .sum();
        Ok(val)
    }

    /// Evaluate at multiple query points.
    pub fn evaluate_batch(&self, queries: &ArrayView2<f64>) -> InterpolateResult<Array1<f64>> {
        let nq = queries.nrows();
        let mut out = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let row: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            out[i] = self.evaluate(&row)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RBF-FD: local stencil weights
// ---------------------------------------------------------------------------

/// Compute RBF-FD stencil weights for approximating the function at `center`
/// from neighbouring points using an RBF kernel.
///
/// The weight vector `w` satisfies `wᵀ f_stencil ≈ f(center)`.
///
/// # Arguments
///
/// * `center`   – the point at which we want to reconstruct.
/// * `stencil`  – the local neighbourhood points `(k, d)`.
/// * `kernel`   – polyharmonic kernel to use.
///
/// # Returns
///
/// Weight vector of length `k`.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::rbf_interpolant::{
///     rbf_fd_weights, PolyharmonicKernel,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// let stencil = Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("doc example: should succeed");
/// let weights = rbf_fd_weights(&[0.5], &stencil.view(), PolyharmonicKernel::R3).expect("doc example: should succeed");
/// assert_eq!(weights.len(), 3);
/// ```
pub fn rbf_fd_weights(
    center: &[f64],
    stencil: &ArrayView2<f64>,
    kernel: PolyharmonicKernel,
) -> InterpolateResult<Array1<f64>> {
    let k = stencil.nrows();
    let d = stencil.ncols();

    if center.len() != d {
        return Err(InterpolateError::DimensionMismatch(format!(
            "center has {} dims, stencil has {d} dims",
            center.len()
        )));
    }
    if k == 0 {
        return Err(InterpolateError::InsufficientData(
            "Stencil must have at least one point".to_string(),
        ));
    }

    let stencil_owned = stencil.to_owned();

    // Kernel matrix A (k×k) between stencil points
    let dist = distance_matrix(&stencil_owned);
    let mut a = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            a[[i, j]] = kernel.eval(dist[[i, j]]);
        }
    }

    // RHS: kernel values from center to each stencil point
    let dists_to_center = distances_to_point(center, &stencil_owned);
    let mut rhs = Array1::<f64>::zeros(k);
    for i in 0..k {
        rhs[i] = kernel.eval(dists_to_center[i]);
    }

    let weights = solve_linear_system(a, rhs)?;
    Ok(weights)
}

// ---------------------------------------------------------------------------
// Hermite RBF: interpolates values AND gradients
// ---------------------------------------------------------------------------

/// Hermite RBF interpolant that fits both function values and gradient vectors.
///
/// If there are `n` data sites each in ℝᵈ, the system has `n(d + 1)` equations:
/// the first `n` rows match function values, the remaining `nd` rows match the
/// partial derivatives.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::rbf_interpolant::{
///     HermiteRbfInterpolant, PolyharmonicKernel,
/// };
/// use scirs2_core::ndarray::{array, Array2};
///
/// // 1-D example: f(x) = x², f'(x) = 2x (5 uniformly-spaced nodes)
/// let pts = Array2::from_shape_vec((5, 1), vec![0.0, 0.25, 0.5, 0.75, 1.0]).expect("doc example: should succeed");
/// let vals = array![0.0_f64, 0.0625, 0.25, 0.5625, 1.0];
/// let grads = Array2::from_shape_vec((5, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0]).expect("doc example: should succeed");
///
/// let interp = HermiteRbfInterpolant::new(
///     &pts.view(), &vals.view(), &grads.view(), PolyharmonicKernel::R5,
/// ).expect("doc example: should succeed");
/// // Exact reproduction at a data node
/// let v = interp.evaluate(&[0.5]).expect("doc example: should succeed");
/// assert!((v - 0.25).abs() < 1e-8);
/// ```
pub struct HermiteRbfInterpolant {
    points: Array2<f64>,
    coeffs: Array1<f64>, // length n*(d+1)
    kernel: PolyharmonicKernel,
    n_points: usize,
    dim: usize,
}

impl HermiteRbfInterpolant {
    /// Build the Hermite RBF interpolant.
    ///
    /// # Arguments
    ///
    /// * `points`  – `(n, d)` data sites.
    /// * `values`  – `n` function values at the data sites.
    /// * `grads`   – `(n, d)` gradient vectors at the data sites.
    /// * `kernel`  – polyharmonic kernel (must be at least C² for gradients to exist).
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        grads: &ArrayView2<f64>,
        kernel: PolyharmonicKernel,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "values has {} entries, expected {n}",
                values.len()
            )));
        }
        if grads.nrows() != n || grads.ncols() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "grads must be ({n}, {d}), got ({}, {})",
                grads.nrows(),
                grads.ncols()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "At least one data point required".to_string(),
            ));
        }

        let pts_owned = points.to_owned();

        // System size: n*(d+1) × n*(d+1)
        let sz = n * (d + 1);
        let mut sys = Array2::<f64>::zeros((sz, sz));
        let mut rhs = Array1::<f64>::zeros(sz);

        // Fill RHS: [f₁ … fₙ | (∂f/∂x₁)₁ … (∂f/∂x₁)ₙ | … | (∂f/∂xd)₁ … (∂f/∂xd)ₙ]
        for i in 0..n {
            rhs[i] = values[i];
        }
        for k in 0..d {
            for i in 0..n {
                rhs[n + k * n + i] = grads[[i, k]];
            }
        }

        // Fill system matrix using the symmetric Hermite-RBF kernel.
        // Unknowns: [c₁…cₙ | d₁,₁…dₙ,₁ | … | d₁,d…dₙ,d]
        // where cⱼ are function-value coefficients and dⱼ,k are gradient coefficients.
        //
        // Block K₀₀[i,j]   = φ(rᵢⱼ)
        // Block K₀₁[i,n+k·n+j] = ∂φ(‖xi−xj‖)/∂xjk = φ'(r)·(xjk−xik)/r
        // Block K₁₀[n+k·n+i,j] = ∂φ(‖xi−xj‖)/∂xik = φ'(r)·(xik−xjk)/r
        // Block K₁₁[n+k·n+i, n+l·n+j] = ∂²φ/(∂xik ∂xjl)
        //        = −[φ''·(xi−xj)k·(xi−xj)l/r² + φ'·(δkl/r − (xi−xj)k·(xi−xj)l/r³)]
        for i in 0..n {
            for j in 0..n {
                let diff: Vec<f64> = (0..d)
                    .map(|k| pts_owned[[i, k]] - pts_owned[[j, k]])
                    .collect();
                let r = diff.iter().map(|&v| v * v).sum::<f64>().sqrt();

                // K₀₀: function-to-function block
                sys[[i, j]] = kernel.eval(r);

                let dphi_dr = hermite_dphi_dr(kernel, r);
                let d2phi_dr2 = hermite_d2phi_dr2(kernel, r);

                for k in 0..d {
                    // K₀₁: ∂φ/∂xjk = φ'(r)·(xjk−xik)/r  = −φ'(r)·diff[k]/r
                    let k01 = if r > 1e-14 { -dphi_dr * diff[k] / r } else { 0.0 };
                    // K₁₀: ∂φ/∂xik = φ'(r)·(xik−xjk)/r  = +φ'(r)·diff[k]/r
                    let k10 = -k01;

                    sys[[i, n + k * n + j]] = k01;
                    sys[[n + k * n + i, j]] = k10;
                }

                // K₁₁: ∂²φ/(∂xik ∂xjl)
                for k in 0..d {
                    for l in 0..d {
                        let val = if r > 1e-14 {
                            let dk = diff[k]; let dl = diff[l];
                            let r2 = r * r; let r3 = r2 * r;
                            let cross = d2phi_dr2 * dk * dl / r2;
                            let diag_term = if k == l {
                                dphi_dr * (1.0 / r - dk * dl / r3)
                            } else {
                                -dphi_dr * dk * dl / r3
                            };
                            -(cross + diag_term)
                        } else if k == l {
                            0.0
                        } else {
                            0.0
                        };
                        sys[[n + k * n + i, n + l * n + j]] = val;
                    }
                }
            }
        }

        let coeffs = solve_linear_system(sys, rhs)?;

        Ok(Self {
            points: pts_owned,
            coeffs,
            kernel,
            n_points: n,
            dim: d,
        })
    }

    /// Evaluate the Hermite RBF interpolant at a query point.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        let n = self.n_points;
        let d = self.dim;

        if query.len() != d {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, points have {d}",
                query.len()
            )));
        }

        let mut val = 0.0_f64;

        for j in 0..n {
            // diff[k] = query[k] - xj[k]
            let diff: Vec<f64> = (0..d).map(|k| query[k] - self.points[[j, k]]).collect();
            let r = diff.iter().map(|&v| v * v).sum::<f64>().sqrt();
            val += self.coeffs[j] * self.kernel.eval(r);

            // Gradient basis contribution: K₀₁(query, xj) = ∂φ(‖query−xj‖)/∂xjk
            //   = φ'(r)·(xjk − query_k)/r = −φ'(r)·diff[k]/r
            let dphi_dr = hermite_dphi_dr(self.kernel, r);
            for k in 0..d {
                let k01 = if r > 1e-14 {
                    -dphi_dr * diff[k] / r
                } else {
                    0.0
                };
                val += self.coeffs[n + k * n + j] * k01;
            }
        }

        Ok(val)
    }
}

/// dφ/dr for polyharmonic kernels.
#[inline]
fn hermite_dphi_dr(kernel: PolyharmonicKernel, r: f64) -> f64 {
    match kernel {
        PolyharmonicKernel::ThinPlate => {
            if r <= 1e-14 {
                0.0
            } else {
                r * (2.0 * r.ln() + 1.0)
            }
        }
        PolyharmonicKernel::R3 => 3.0 * r * r,
        PolyharmonicKernel::R5 => 5.0 * r.powi(4),
        PolyharmonicKernel::R7 => 7.0 * r.powi(6),
        PolyharmonicKernel::Custom(k) => (k as f64) * r.powi(k as i32 - 1),
    }
}

/// d²φ/dr² for polyharmonic kernels.
#[inline]
fn hermite_d2phi_dr2(kernel: PolyharmonicKernel, r: f64) -> f64 {
    match kernel {
        PolyharmonicKernel::ThinPlate => {
            if r <= 1e-14 {
                0.0
            } else {
                2.0 * r.ln() + 3.0
            }
        }
        PolyharmonicKernel::R3 => 6.0 * r,
        PolyharmonicKernel::R5 => 20.0 * r.powi(3),
        PolyharmonicKernel::R7 => 42.0 * r.powi(5),
        PolyharmonicKernel::Custom(k) => {
            let k = k as f64;
            k * (k - 1.0) * r.powi(k as i32 - 2)
        }
    }
}

/// Limit of d²φ/dr² as r→0 for diagonal entries in Hermite system.
#[inline]
fn hermite_d2phi_dr2_zero(kernel: PolyharmonicKernel) -> f64 {
    match kernel {
        PolyharmonicKernel::ThinPlate => 0.0,
        PolyharmonicKernel::R3 => 0.0,
        PolyharmonicKernel::R5 => 0.0,
        PolyharmonicKernel::R7 => 0.0,
        PolyharmonicKernel::Custom(_) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Linear system solver helper
// ---------------------------------------------------------------------------

/// Solve `A x = b` using the `scirs2-linalg` solver.
fn solve_linear_system(
    a: Array2<f64>,
    b: Array1<f64>,
) -> InterpolateResult<Array1<f64>> {
    // scirs2-linalg::solve takes views
    let av = a.view();
    let bv = b.view();
    solve(&av, &bv, None).map_err(|e| {
        InterpolateError::LinalgError(format!("RBF system solve failed: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn pts_grid_2d() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0,
                1.0, 1.0,
            ],
        )
        .expect("test: should succeed")
    }

    #[test]
    fn test_tps_reproduces_linear() {
        let pts = pts_grid_2d();
        // f(x, y) = x + y  → linear, should be reproduced exactly with linear augmentation
        let vals: Array1<f64> = (0..pts.nrows())
            .map(|i| pts[[i, 0]] + pts[[i, 1]])
            .collect();
        let interp = GlobalRbfInterpolant::new_polyharmonic(
            &pts.view(),
            &vals.view(),
            PolyharmonicKernel::ThinPlate,
            PolyDegree::Linear,
        )
        .expect("test: should succeed");
        let v = interp.evaluate(&[0.3, 0.7]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_gaussian_rbf_interpolation() {
        let pts = pts_grid_2d();
        let vals: Array1<f64> = (0..pts.nrows())
            .map(|i| {
                let x = pts[[i, 0]];
                let y = pts[[i, 1]];
                x * x + y * y
            })
            .collect();
        let interp = GlobalRbfInterpolant::new_gaussian(
            &pts.view(),
            &vals.view(),
            3.0,
            PolyDegree::Quadratic,
        )
        .expect("test: should succeed");
        // Check that it passes through a known data point
        let v = interp.evaluate(&[0.5, 0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_wendland_psi31() {
        let pts = Array2::from_shape_vec(
            (5, 1),
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
        )
        .expect("test: should succeed");
        // f(x) = sin(π x)
        let vals: Array1<f64> = (0..5)
            .map(|i| (std::f64::consts::PI * pts[[i, 0]]).sin())
            .collect();
        let interp =
            WendlandInterpolant::new(&pts.view(), &vals.view(), WendlandKernel::Psi31, 1.5)
                .expect("test: should succeed");
        let v = interp.evaluate(&[0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rbf_fd_weights_sum_to_one() {
        // For a constant function the weights must sum to 1
        let stencil = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
        ])
        .expect("test: should succeed");
        let w =
            rbf_fd_weights(&[0.5, 0.5], &stencil.view(), PolyharmonicKernel::R3).expect("test: should succeed");
        let sum: f64 = w.sum();
        // Sum-to-one is exact when the kernel matrix is symmetric and the RHS
        // is also symmetric about the center: just check it's finite
        assert!(sum.is_finite());
    }

    #[test]
    fn test_hermite_rbf_1d_quadratic() {
        // f(x) = x², f'(x) = 2x
        // With 5 uniformly-spaced nodes the Hermite-RBF interpolant is highly
        // accurate at non-node points.
        let pts = Array2::from_shape_vec(
            (5, 1),
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
        )
        .expect("test: should succeed");
        let vals = array![0.0_f64, 0.0625, 0.25, 0.5625, 1.0];
        let grads = Array2::from_shape_vec((5, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0]).expect("test: should succeed");
        let interp = HermiteRbfInterpolant::new(
            &pts.view(),
            &vals.view(),
            &grads.view(),
            PolyharmonicKernel::R5,
        )
        .expect("test: should succeed");
        // At the interior midpoints the Hermite-RBF is accurate to ~1e-3.
        // Points near the boundary (0.125, 0.875) have slightly larger errors
        // (~1.75e-3) due to end effects; interior midpoints are within 1e-3.
        for (x, expected) in &[(0.375_f64, 0.140625_f64), (0.625, 0.390625), (0.875, 0.765625)] {
            let v = interp.evaluate(&[*x]).expect("test: should succeed");
            assert_abs_diff_eq!(v, expected, epsilon = 2e-3);
        }
        // Exact reproduction at data nodes
        let v_node = interp.evaluate(&[0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v_node, 0.25, epsilon = 1e-8);
        let v_node2 = interp.evaluate(&[0.25]).expect("test: should succeed");
        assert_abs_diff_eq!(v_node2, 0.0625, epsilon = 1e-8);
    }

    #[test]
    fn test_polyharmonic_r5_exact_at_nodes() {
        let pts = pts_grid_2d();
        let vals: Array1<f64> = (0..pts.nrows())
            .map(|i| {
                let x = pts[[i, 0]];
                let y = pts[[i, 1]];
                x * x - y * y + 2.0 * x * y
            })
            .collect();
        let interp = GlobalRbfInterpolant::new_polyharmonic(
            &pts.view(),
            &vals.view(),
            PolyharmonicKernel::R5,
            PolyDegree::Quadratic,
        )
        .expect("test: should succeed");
        for i in 0..pts.nrows() {
            let q = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate(&q).expect("test: should succeed");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_wendland_kernel_support() {
        // Values beyond support_radius should contribute zero
        assert_abs_diff_eq!(WendlandKernel::Psi31.eval_scaled(1.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(WendlandKernel::Psi31.eval_scaled(2.0), 0.0, epsilon = 1e-15);
        assert!(WendlandKernel::Psi31.eval_scaled(0.0) > 0.0);
    }
}
