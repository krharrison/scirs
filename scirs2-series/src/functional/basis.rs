//! Basis function systems for functional data analysis
//!
//! This module provides various basis function systems used to represent
//! functional data as linear combinations of basis functions.
//!
//! # Available Basis Systems
//!
//! - [`BSplineBasis`]: B-spline basis with arbitrary knot sequences
//! - [`FourierBasis`]: Truncated Fourier basis (sine/cosine pairs)
//! - [`WaveletBasis`]: Haar and Daubechies D4 wavelet basis functions
//! - [`MonomialBasis`]: Polynomial (monomial) basis functions

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};

// ============================================================
// Core Trait
// ============================================================

/// A basis function system over a compact domain
///
/// Implementors provide evaluation of basis functions and their derivatives
/// at arbitrary points, enabling smooth functional data representations.
pub trait BasisSystem: Send + Sync {
    /// Number of basis functions in the system
    fn n_basis(&self) -> usize;

    /// Evaluate all basis functions at a single point `t`
    ///
    /// Returns a vector of length `n_basis()`.
    fn evaluate(&self, t: f64) -> Result<Array1<f64>>;

    /// Evaluate the `order`-th derivative of all basis functions at `t`
    fn evaluate_deriv(&self, t: f64, order: usize) -> Result<Array1<f64>>;

    /// Gram matrix G where G[i,j] = ∫ φ_i(t) φ_j(t) dt
    fn gram_matrix(&self) -> Result<Array2<f64>>;

    /// Penalty matrix D for `order`-th derivative roughness:
    /// D[i,j] = ∫ φ_i^(order)(t) φ_j^(order)(t) dt
    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>>;
}

/// Evaluate all basis functions at multiple data points, yielding an (n_points × n_basis) matrix
pub fn evaluate_basis_matrix<B: BasisSystem>(
    basis: &B,
    points: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n = points.len();
    let k = basis.n_basis();
    let mut mat = Array2::zeros((n, k));
    for (i, &t) in points.iter().enumerate() {
        let row = basis.evaluate(t)?;
        for j in 0..k {
            mat[[i, j]] = row[j];
        }
    }
    Ok(mat)
}

/// Evaluate derivatives of all basis functions at multiple data points
pub fn evaluate_deriv_matrix<B: BasisSystem>(
    basis: &B,
    points: &Array1<f64>,
    order: usize,
) -> Result<Array2<f64>> {
    let n = points.len();
    let k = basis.n_basis();
    let mut mat = Array2::zeros((n, k));
    for (i, &t) in points.iter().enumerate() {
        let row = basis.evaluate_deriv(t, order)?;
        for j in 0..k {
            mat[[i, j]] = row[j];
        }
    }
    Ok(mat)
}

// ============================================================
// B-Spline Basis
// ============================================================

/// B-spline basis defined by a knot vector and spline order
///
/// Uses the Cox–de Boor recursion to evaluate B-splines of any order.
/// The domain is `[knots[order-1], knots[knots.len()-order]]`.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    /// Extended knot vector (including repeated boundary knots)
    pub knots: Vec<f64>,
    /// Spline order k (degree = k-1); must be >= 1
    pub order: usize,
    /// Number of basis functions = len(knots) - order
    n_basis: usize,
    /// Left boundary of active domain
    domain_min: f64,
    /// Right boundary of active domain
    domain_max: f64,
    /// Number of quadrature points for Gram matrix integration
    n_quad: usize,
}

impl BSplineBasis {
    /// Construct a B-spline basis with the given (extended) knot vector and order
    ///
    /// The knot vector must be non-decreasing and must have length >= 2*order.
    pub fn new(knots: Vec<f64>, order: usize) -> Result<Self> {
        if order == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "B-spline order must be >= 1".to_string(),
            ));
        }
        if knots.len() < 2 * order {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Knot vector length {} must be >= 2*order = {}",
                knots.len(),
                2 * order
            )));
        }
        // Verify non-decreasing
        for i in 1..knots.len() {
            if knots[i] < knots[i - 1] {
                return Err(TimeSeriesError::InvalidInput(
                    "Knot vector must be non-decreasing".to_string(),
                ));
            }
        }
        let n_basis = knots.len() - order;
        let domain_min = knots[order - 1];
        let domain_max = knots[knots.len() - order];
        Ok(Self {
            knots,
            order,
            n_basis,
            domain_min,
            domain_max,
            n_quad: 200,
        })
    }

    /// Construct a uniform B-spline basis on [0,1] with `n_interior` interior knots
    pub fn uniform(n_interior: usize, order: usize) -> Result<Self> {
        let n_knots = n_interior + 2 * order;
        let mut knots = Vec::with_capacity(n_knots);
        // Repeat boundary knots `order` times
        for _ in 0..order {
            knots.push(0.0);
        }
        for i in 1..=(n_interior) {
            knots.push(i as f64 / (n_interior + 1) as f64);
        }
        for _ in 0..order {
            knots.push(1.0);
        }
        Self::new(knots, order)
    }

    /// Cox–de Boor recursion: evaluate B_{i,k}(t)
    fn bspline_value(&self, i: usize, k: usize, t: f64) -> f64 {
        if k == 1 {
            let left = self.knots[i];
            let right = self.knots[i + 1];
            if t >= left && t < right {
                return 1.0;
            }
            // Special handling for last knot span (closed on right)
            if t >= right && (i + 1) == self.knots.len() - 1 {
                return 1.0;
            }
            return 0.0;
        }
        let mut result = 0.0;
        let denom1 = self.knots[i + k - 1] - self.knots[i];
        if denom1 > 0.0 {
            result += ((t - self.knots[i]) / denom1) * self.bspline_value(i, k - 1, t);
        }
        let denom2 = self.knots[i + k] - self.knots[i + 1];
        if denom2 > 0.0 {
            result +=
                ((self.knots[i + k] - t) / denom2) * self.bspline_value(i + 1, k - 1, t);
        }
        result
    }

    /// Evaluate derivative of B_{i,k}(t) of given order using recursion
    fn bspline_deriv(&self, i: usize, k: usize, t: f64, deriv_order: usize) -> f64 {
        if deriv_order == 0 {
            return self.bspline_value(i, k, t);
        }
        if k <= 1 {
            return 0.0;
        }
        let mut result = 0.0;
        let d1 = self.knots[i + k - 1] - self.knots[i];
        if d1 > 0.0 {
            result += (k as f64 - 1.0) / d1
                * self.bspline_deriv(i, k - 1, t, deriv_order - 1);
        }
        let d2 = self.knots[i + k] - self.knots[i + 1];
        if d2 > 0.0 {
            result -= (k as f64 - 1.0) / d2
                * self.bspline_deriv(i + 1, k - 1, t, deriv_order - 1);
        }
        result
    }

    fn gauss_legendre_points(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        // Generate Gauss-Legendre quadrature on [domain_min, domain_max]
        let (nodes_01, weights_01) = gauss_legendre_01(n);
        let a = self.domain_min;
        let b = self.domain_max;
        let scale = b - a;
        let nodes: Vec<f64> = nodes_01.iter().map(|&x| a + scale * x).collect();
        let weights: Vec<f64> = weights_01.iter().map(|&w| scale * w).collect();
        (nodes, weights)
    }
}

impl BasisSystem for BSplineBasis {
    fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn evaluate(&self, t: f64) -> Result<Array1<f64>> {
        let mut vals = Array1::zeros(self.n_basis);
        for i in 0..self.n_basis {
            vals[i] = self.bspline_value(i, self.order, t);
        }
        Ok(vals)
    }

    fn evaluate_deriv(&self, t: f64, order: usize) -> Result<Array1<f64>> {
        let mut vals = Array1::zeros(self.n_basis);
        for i in 0..self.n_basis {
            vals[i] = self.bspline_deriv(i, self.order, t, order);
        }
        Ok(vals)
    }

    fn gram_matrix(&self) -> Result<Array2<f64>> {
        let (nodes, weights) = self.gauss_legendre_points(self.n_quad);
        let k = self.n_basis;
        let mut g = Array2::zeros((k, k));
        for (&t, &w) in nodes.iter().zip(weights.iter()) {
            let phi = self.evaluate(t)?;
            for i in 0..k {
                for j in 0..=i {
                    let val = w * phi[i] * phi[j];
                    g[[i, j]] += val;
                    if i != j {
                        g[[j, i]] += val;
                    }
                }
            }
        }
        Ok(g)
    }

    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>> {
        let (nodes, weights) = self.gauss_legendre_points(self.n_quad);
        let k = self.n_basis;
        let mut d = Array2::zeros((k, k));
        for (&t, &w) in nodes.iter().zip(weights.iter()) {
            let dphi = self.evaluate_deriv(t, order)?;
            for i in 0..k {
                for j in 0..=i {
                    let val = w * dphi[i] * dphi[j];
                    d[[i, j]] += val;
                    if i != j {
                        d[[j, i]] += val;
                    }
                }
            }
        }
        Ok(d)
    }
}

// ============================================================
// Fourier Basis
// ============================================================

/// Truncated Fourier basis on [0, period]
///
/// The basis consists of the constant 1 followed by sin/cos pairs:
/// `φ_0(t) = 1`, `φ_{2k-1}(t) = sin(2πkt/T)`, `φ_{2k}(t) = cos(2πkt/T)`
/// for k = 1, 2, ..., n_harmonics.
/// Total basis size = 2*n_harmonics + 1.
#[derive(Debug, Clone)]
pub struct FourierBasis {
    /// Number of harmonic pairs (sin/cos)
    pub n_harmonics: usize,
    /// Period T
    pub period: f64,
    /// Number of quadrature points for integration
    n_quad: usize,
}

impl FourierBasis {
    /// Create a Fourier basis with the given number of harmonics and period
    pub fn new(n_harmonics: usize, period: f64) -> Result<Self> {
        if period <= 0.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Fourier basis period must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_harmonics,
            period,
            n_quad: 400,
        })
    }

    /// Evaluate the j-th basis function at t
    fn eval_one(&self, j: usize, t: f64) -> f64 {
        if j == 0 {
            return 1.0;
        }
        let k = (j + 1) / 2;
        let arg = 2.0 * std::f64::consts::PI * k as f64 * t / self.period;
        if j % 2 == 1 {
            arg.sin()
        } else {
            arg.cos()
        }
    }

    /// Evaluate the `deriv_order`-th derivative of basis function j at t
    fn eval_deriv_one(&self, j: usize, t: f64, deriv_order: usize) -> f64 {
        if j == 0 {
            if deriv_order == 0 {
                return 1.0;
            }
            return 0.0;
        }
        let k = (j + 1) / 2;
        let omega = 2.0 * std::f64::consts::PI * k as f64 / self.period;
        let arg = omega * t;
        // Derivative of order d of A*sin(ωt) = A*ω^d * sin(ωt + d*π/2)
        let phase_shift = deriv_order as f64 * std::f64::consts::FRAC_PI_2;
        let amplitude = omega.powi(deriv_order as i32);
        if j % 2 == 1 {
            amplitude * (arg + phase_shift).sin()
        } else {
            amplitude * (arg + phase_shift).cos()
        }
    }
}

impl BasisSystem for FourierBasis {
    fn n_basis(&self) -> usize {
        2 * self.n_harmonics + 1
    }

    fn evaluate(&self, t: f64) -> Result<Array1<f64>> {
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        for j in 0..k {
            vals[j] = self.eval_one(j, t);
        }
        Ok(vals)
    }

    fn evaluate_deriv(&self, t: f64, order: usize) -> Result<Array1<f64>> {
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        for j in 0..k {
            vals[j] = self.eval_deriv_one(j, t, order);
        }
        Ok(vals)
    }

    fn gram_matrix(&self) -> Result<Array2<f64>> {
        // Fourier basis is orthogonal on [0, period]:
        // <1,1> = period, <sin,sin> = period/2, <cos,cos> = period/2, cross = 0
        let k = self.n_basis();
        let mut g = Array2::zeros((k, k));
        g[[0, 0]] = self.period;
        for i in 1..k {
            g[[i, i]] = self.period / 2.0;
        }
        Ok(g)
    }

    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>> {
        // Analytical formula: integral of (φ'_j)^2 over [0,period]
        let k = self.n_basis();
        let mut d = Array2::zeros((k, k));
        if order == 0 {
            return self.gram_matrix();
        }
        // Constant function: derivative = 0
        // For sin/cos with frequency k: |d^n/dt^n φ_j|^2 integral = (ω_k)^{2n} * period/2
        for j in 1..k {
            let freq = (j + 1) / 2;
            let omega = 2.0 * std::f64::consts::PI * freq as f64 / self.period;
            d[[j, j]] = omega.powi(2 * order as i32) * self.period / 2.0;
        }
        Ok(d)
    }
}

// ============================================================
// Wavelet Basis
// ============================================================

/// Type of wavelet used in WaveletBasis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Haar wavelet (piecewise constant)
    Haar,
    /// Daubechies D4 wavelet (4-tap filter)
    Daubechies4,
}

/// Wavelet basis using Haar or Daubechies D4 wavelets
///
/// Provides a multi-resolution basis on [0, 1] at `n_levels` resolution levels.
/// For Haar wavelets the basis functions are piecewise constant.
/// For D4 the scaling function and wavelets are evaluated via the cascade algorithm.
#[derive(Debug, Clone)]
pub struct WaveletBasis {
    /// Wavelet type
    pub wavelet_type: WaveletType,
    /// Number of resolution levels
    pub n_levels: usize,
    /// Number of cascade iterations for D4 evaluation
    pub cascade_iters: usize,
}

impl WaveletBasis {
    /// Create a wavelet basis with the given type and number of resolution levels
    pub fn new(wavelet_type: WaveletType, n_levels: usize) -> Result<Self> {
        if n_levels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "WaveletBasis requires at least 1 level".to_string(),
            ));
        }
        Ok(Self {
            wavelet_type,
            n_levels,
            cascade_iters: 8,
        })
    }

    /// Haar scaling function at resolution j, translation k: ψ_{j,k}(t) = 2^{j/2} h(2^j t - k)
    fn haar_scaling(t: f64, j: usize, k: usize) -> f64 {
        let scale = (1u64 << j) as f64;
        let u = scale * t - k as f64;
        if u >= 0.0 && u < 1.0 {
            scale.sqrt()
        } else {
            0.0
        }
    }

    /// Haar mother wavelet at resolution j, translation k
    fn haar_wavelet(t: f64, j: usize, k: usize) -> f64 {
        let scale = (1u64 << j) as f64;
        let u = scale * t - k as f64;
        let amplitude = scale.sqrt();
        if u >= 0.0 && u < 0.5 {
            amplitude
        } else if u >= 0.5 && u < 1.0 {
            -amplitude
        } else {
            0.0
        }
    }

    /// D4 low-pass filter coefficients (Daubechies 4)
    fn d4_lo() -> [f64; 4] {
        let s3 = 3.0_f64.sqrt();
        let denom = 4.0 * 2.0_f64.sqrt();
        [
            (1.0 + s3) / denom,
            (3.0 + s3) / denom,
            (3.0 - s3) / denom,
            (1.0 - s3) / denom,
        ]
    }

    /// D4 high-pass (wavelet) filter coefficients
    fn d4_hi() -> [f64; 4] {
        let lo = Self::d4_lo();
        [-lo[3], lo[2], -lo[1], lo[0]]
    }

    /// Evaluate D4 scaling function at t ∈ [0,1] via cascade algorithm
    fn d4_scaling(&self, t: f64) -> f64 {
        if t < 0.0 || t > 1.0 {
            return 0.0;
        }
        // Start with box function on [0,1], iterate cascade
        let n_points = 1 << self.cascade_iters;
        let idx = (t * n_points as f64) as usize;
        let idx = idx.min(n_points - 1);
        // Evaluate at discrete points using cascade
        let values = self.cascade_scaling(self.cascade_iters);
        if idx < values.len() {
            values[idx]
        } else {
            0.0
        }
    }

    /// Cascade algorithm: produce scaling function values at 2^n evenly-spaced points on [0, support]
    fn cascade_scaling(&self, n: usize) -> Vec<f64> {
        let lo = Self::d4_lo();
        let support = 3; // D4 support is [0, 3]
        let n_out = support * (1 << n);
        // Initialize with delta at 0
        let mut vals = vec![0.0_f64; n_out + 1];
        vals[0] = 1.0;
        for _iter in 0..n {
            let len = vals.len();
            let mut new_vals = vec![0.0_f64; 2 * len];
            for i in 0..len {
                for (k, &h) in lo.iter().enumerate() {
                    let idx = 2 * i + k;
                    if idx < new_vals.len() {
                        new_vals[idx] += h * vals[i] * std::f64::consts::SQRT_2;
                    }
                }
            }
            vals = new_vals;
        }
        vals
    }

    /// Evaluate D4 wavelet function at t ∈ [0,1]
    fn d4_wavelet(&self, t: f64) -> f64 {
        if t < 0.0 || t > 1.0 {
            return 0.0;
        }
        let hi = Self::d4_hi();
        let n_points = 1 << self.cascade_iters;
        let scaling = self.cascade_scaling(self.cascade_iters);
        let support_n = scaling.len();
        // ψ(t) = √2 Σ g_k φ(2t - k)
        let mut result = 0.0;
        for (k, &g) in hi.iter().enumerate() {
            let u = 2.0 * t - k as f64;
            if u >= 0.0 && u <= 3.0 {
                let idx = (u * n_points as f64) as usize;
                let idx = idx.min(support_n - 1);
                result += std::f64::consts::SQRT_2 * g * scaling[idx];
            }
        }
        result
    }

    fn eval_basis_one(&self, j_idx: usize, t: f64) -> f64 {
        match self.wavelet_type {
            WaveletType::Haar => {
                if j_idx == 0 {
                    // Scaling function at level 0 (constant 1)
                    return if t >= 0.0 && t <= 1.0 { 1.0 } else { 0.0 };
                }
                // Enumerate wavelet functions: level j has 2^j wavelets
                let mut count = 1usize;
                let mut level = 0usize;
                loop {
                    let n_in_level = 1usize << level;
                    if j_idx < count + n_in_level {
                        let k = j_idx - count;
                        return Self::haar_wavelet(t, level, k);
                    }
                    count += n_in_level;
                    level += 1;
                    if level > self.n_levels + 2 {
                        return 0.0;
                    }
                }
            }
            WaveletType::Daubechies4 => {
                if j_idx == 0 {
                    return self.d4_scaling(t);
                }
                // Enumerate wavelet functions at various levels
                let mut count = 1usize;
                let mut level = 0usize;
                loop {
                    let n_in_level = 1usize << level;
                    if j_idx < count + n_in_level {
                        let k = j_idx - count;
                        let scale = (1u64 << level) as f64;
                        let u = scale * t - k as f64;
                        return scale.sqrt() * self.d4_wavelet(u);
                    }
                    count += n_in_level;
                    level += 1;
                    if level > self.n_levels + 2 {
                        return 0.0;
                    }
                }
            }
        }
    }
}

impl BasisSystem for WaveletBasis {
    fn n_basis(&self) -> usize {
        // 1 scaling function + sum_{j=0}^{n_levels-1} 2^j wavelets
        1 + (1usize << self.n_levels) - 1
    }

    fn evaluate(&self, t: f64) -> Result<Array1<f64>> {
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        for j in 0..k {
            vals[j] = self.eval_basis_one(j, t);
        }
        Ok(vals)
    }

    fn evaluate_deriv(&self, t: f64, order: usize) -> Result<Array1<f64>> {
        if order == 0 {
            return self.evaluate(t);
        }
        // Wavelet derivatives via finite differences (5-point stencil)
        let h = 1e-5;
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        if order == 1 {
            let phi_plus = self.evaluate(t + h)?;
            let phi_minus = self.evaluate(t - h)?;
            for j in 0..k {
                vals[j] = (phi_plus[j] - phi_minus[j]) / (2.0 * h);
            }
        } else if order == 2 {
            let phi_plus = self.evaluate(t + h)?;
            let phi_center = self.evaluate(t)?;
            let phi_minus = self.evaluate(t - h)?;
            for j in 0..k {
                vals[j] = (phi_plus[j] - 2.0 * phi_center[j] + phi_minus[j]) / (h * h);
            }
        } else {
            // Higher orders via recursive finite differences
            let phi1 = self.evaluate_deriv(t + h, order - 1)?;
            let phi2 = self.evaluate_deriv(t - h, order - 1)?;
            for j in 0..k {
                vals[j] = (phi1[j] - phi2[j]) / (2.0 * h);
            }
        }
        Ok(vals)
    }

    fn gram_matrix(&self) -> Result<Array2<f64>> {
        let n = 400;
        let (nodes, weights) = gauss_legendre_01_scaled(n, 0.0, 1.0);
        let k = self.n_basis();
        let mut g = Array2::zeros((k, k));
        for (&t, &w) in nodes.iter().zip(weights.iter()) {
            let phi = self.evaluate(t)?;
            for i in 0..k {
                for j in 0..=i {
                    let val = w * phi[i] * phi[j];
                    g[[i, j]] += val;
                    if i != j {
                        g[[j, i]] += val;
                    }
                }
            }
        }
        Ok(g)
    }

    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>> {
        let n = 400;
        let (nodes, weights) = gauss_legendre_01_scaled(n, 0.0, 1.0);
        let k = self.n_basis();
        let mut d = Array2::zeros((k, k));
        for (&t, &w) in nodes.iter().zip(weights.iter()) {
            let dphi = self.evaluate_deriv(t, order)?;
            for i in 0..k {
                for j in 0..=i {
                    let val = w * dphi[i] * dphi[j];
                    d[[i, j]] += val;
                    if i != j {
                        d[[j, i]] += val;
                    }
                }
            }
        }
        Ok(d)
    }
}

// ============================================================
// Monomial (Polynomial) Basis
// ============================================================

/// Polynomial (monomial) basis φ_j(t) = t^j for j = 0, 1, ..., degree
///
/// Defined on the interval [domain_min, domain_max].
/// Note: for high-degree polynomials, consider using an orthogonal polynomial
/// basis (e.g., Legendre) for numerical stability.
#[derive(Debug, Clone)]
pub struct MonomialBasis {
    /// Polynomial degree (basis size = degree + 1)
    pub degree: usize,
    /// Left endpoint of domain
    pub domain_min: f64,
    /// Right endpoint of domain
    pub domain_max: f64,
}

impl MonomialBasis {
    /// Create a monomial basis on `[domain_min, domain_max]` up to degree `degree`
    pub fn new(degree: usize, domain_min: f64, domain_max: f64) -> Result<Self> {
        if domain_min >= domain_max {
            return Err(TimeSeriesError::InvalidInput(
                "domain_min must be less than domain_max".to_string(),
            ));
        }
        Ok(Self {
            degree,
            domain_min,
            domain_max,
        })
    }

    /// Normalize t to [0, 1]
    fn normalize(&self, t: f64) -> f64 {
        (t - self.domain_min) / (self.domain_max - self.domain_min)
    }

    /// Rising factorial / derivative coefficient: d^k/dt^k [t^j] = j!/(j-k)! * t^(j-k)
    fn poly_deriv_coeff(j: usize, order: usize) -> f64 {
        if order > j {
            return 0.0;
        }
        let mut coeff = 1.0_f64;
        for m in 0..order {
            coeff *= (j - m) as f64;
        }
        coeff
    }
}

impl BasisSystem for MonomialBasis {
    fn n_basis(&self) -> usize {
        self.degree + 1
    }

    fn evaluate(&self, t: f64) -> Result<Array1<f64>> {
        let u = self.normalize(t);
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        let mut pow = 1.0;
        for j in 0..k {
            vals[j] = pow;
            pow *= u;
        }
        Ok(vals)
    }

    fn evaluate_deriv(&self, t: f64, order: usize) -> Result<Array1<f64>> {
        if order == 0 {
            return self.evaluate(t);
        }
        let u = self.normalize(t);
        let scale = self.domain_max - self.domain_min;
        let k = self.n_basis();
        let mut vals = Array1::zeros(k);
        for j in 0..k {
            let coeff = Self::poly_deriv_coeff(j, order);
            if coeff.abs() < 1e-15 {
                vals[j] = 0.0;
            } else {
                let exp = if j >= order { j - order } else { 0 };
                vals[j] = coeff * u.powi(exp as i32) / scale.powi(order as i32);
            }
        }
        Ok(vals)
    }

    fn gram_matrix(&self) -> Result<Array2<f64>> {
        let k = self.n_basis();
        let mut g = Array2::zeros((k, k));
        // Gram matrix on [0,1]: G[i,j] = integral_0^1 u^i u^j du = 1/(i+j+1)
        // Then scale by domain width
        let scale = self.domain_max - self.domain_min;
        for i in 0..k {
            for j in 0..k {
                g[[i, j]] = scale / (i + j + 1) as f64;
            }
        }
        Ok(g)
    }

    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>> {
        if order == 0 {
            return self.gram_matrix();
        }
        let k = self.n_basis();
        let mut d = Array2::zeros((k, k));
        let scale = self.domain_max - self.domain_min;
        // d^order/dt^order [t^j] on [0,1]: coeff_j * u^(j-order)
        // integral_0^1 coeff_i * u^(i-order) * coeff_j * u^(j-order) du = coeff_i*coeff_j / (i+j-2*order+1)
        for i in order..k {
            for j in order..k {
                let ci = Self::poly_deriv_coeff(i, order);
                let cj = Self::poly_deriv_coeff(j, order);
                let exp = (i + j) as i32 - 2 * order as i32;
                if exp >= 0 {
                    d[[i, j]] = ci * cj / (exp as f64 + 1.0)
                        / scale.powi(2 * order as i32 - 1);
                }
            }
        }
        Ok(d)
    }
}

// ============================================================
// Gauss-Legendre quadrature helpers
// ============================================================

/// Gauss-Legendre nodes and weights on [0,1] using n-point rule
/// Uses a simple iterative eigenvalue approach for the Jacobi matrix.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    // Compute nodes/weights on [-1,1] via Golub-Welsch, then map to [0,1]
    let (nodes_sym, weights_sym) = gauss_legendre_sym(n);
    let nodes: Vec<f64> = nodes_sym.iter().map(|&x| (x + 1.0) / 2.0).collect();
    let weights: Vec<f64> = weights_sym.iter().map(|&w| w / 2.0).collect();
    (nodes, weights)
}

/// Gauss-Legendre nodes/weights scaled to [a, b]
pub fn gauss_legendre_01_scaled(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let (nodes_01, weights_01) = gauss_legendre_01(n);
    let scale = b - a;
    let nodes: Vec<f64> = nodes_01.iter().map(|&x| a + scale * x).collect();
    let weights: Vec<f64> = weights_01.iter().map(|&w| scale * w).collect();
    (nodes, weights)
}

/// Gauss-Legendre nodes/weights on [-1,1] via the eigenvalue method (Golub-Welsch)
fn gauss_legendre_sym(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }
    // Off-diagonal elements of Jacobi matrix: beta_i = i / sqrt(4i^2 - 1)
    let mut beta = vec![0.0_f64; n - 1];
    for i in 1..n {
        beta[i - 1] = i as f64 / ((4 * i * i - 1) as f64).sqrt();
    }
    // Compute eigenvalues of symmetric tridiagonal matrix using implicit QR
    let (nodes, eigvecs) = tridiag_eig_sym(n, &beta);
    // Weights = 2 * (first component of eigenvectors)^2
    let weights: Vec<f64> = eigvecs.iter().map(|v| 2.0 * v * v).collect();
    (nodes, weights)
}

/// Compute eigenvalues of symmetric tridiagonal matrix with zero diagonal
/// and off-diagonal `beta`. Returns (eigenvalues, first_components_of_eigenvecs).
fn tridiag_eig_sym(n: usize, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    use std::f64::consts::PI;
    // Initial estimates via Chebyshev nodes
    let mut d = vec![0.0_f64; n]; // diagonal
    let mut e: Vec<f64> = {
        // Extend to length n (the implicit QR algorithm needs indices up to n-1)
        let mut v = beta.to_vec();
        v.push(0.0);
        v
    }; // off-diagonal

    // Store eigenvectors (just first component needed)
    let mut z = vec![1.0_f64; n]; // first component of each eigenvector
    let mut z_full: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut v = vec![0.0_f64; n];
        v[i] = 1.0;
        v
    }).collect();

    let max_iter = 100 * n;
    let eps = f64::EPSILON;

    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find small off-diagonal element
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > max_iter {
                break;
            }
            // Form shift
            let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            let g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });
            let (mut s, mut c, mut p) = (1.0_f64, 1.0_f64, 0.0_f64);
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = (f * f + g * g).sqrt();
                e[i + 1] = r;
                if r.abs() < 1e-300 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                let g_new = d[i + 1] - p;
                let r2 = (d[i] - g_new) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_new + p;
                let g = c * r2 - b;
                // Update eigenvectors
                for k in 0..n {
                    let fv = z_full[k][i + 1];
                    z_full[k][i + 1] = s * z_full[k][i] + c * fv;
                    z_full[k][i] = c * z_full[k][i] - s * fv;
                }
                let _ = g; // suppress unused warning (used next iteration)
                let _ = b;
                let g = g_new + p - r2 * s;
                let _ = g;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    // Extract first component of eigenvectors
    let first_comps: Vec<f64> = (0..n).map(|i| z_full[0][i]).collect();
    (d, first_comps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bspline_partition_of_unity() {
        let basis = BSplineBasis::uniform(5, 4).expect("basis creation failed");
        // B-splines should sum to 1 at interior points
        for &t in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let vals = basis.evaluate(t).expect("evaluate failed");
            let sum: f64 = vals.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fourier_gram_matrix() {
        let basis = FourierBasis::new(3, 1.0).expect("basis creation");
        let g = basis.gram_matrix().expect("gram matrix");
        // Diagonal: period for constant, period/2 for sin/cos
        assert_abs_diff_eq!(g[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[[1, 1]], 0.5, epsilon = 1e-10);
        // Off-diagonal should be zero (orthogonal)
        assert_abs_diff_eq!(g[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_monomial_gram_matrix() {
        let basis = MonomialBasis::new(3, 0.0, 1.0).expect("basis creation");
        let g = basis.gram_matrix().expect("gram matrix");
        // G[0,0] = integral_0^1 1 dt = 1
        assert_abs_diff_eq!(g[[0, 0]], 1.0, epsilon = 1e-10);
        // G[0,1] = integral_0^1 t dt = 0.5
        assert_abs_diff_eq!(g[[0, 1]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_basis_matrix_shape() {
        let basis = BSplineBasis::uniform(3, 4).expect("basis");
        let pts = Array1::from_vec(vec![0.1, 0.2, 0.5, 0.8, 0.9]);
        let mat = evaluate_basis_matrix(&basis, &pts).expect("eval matrix");
        assert_eq!(mat.nrows(), 5);
        assert_eq!(mat.ncols(), basis.n_basis());
    }

    #[test]
    fn test_wavelet_basis_haar() {
        let basis = WaveletBasis::new(WaveletType::Haar, 3).expect("wavelet basis");
        // Should not panic
        let vals = basis.evaluate(0.5).expect("evaluate");
        assert_eq!(vals.len(), basis.n_basis());
    }
}
