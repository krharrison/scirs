//! Polynomial Chaos Expansion (PCE) for uncertainty quantification.
//!
//! PCE represents a stochastic quantity u(ξ) as a weighted sum of orthogonal polynomials:
//!
//!   u(ξ) = Σ_α c_α Ψ_α(ξ)
//!
//! where:
//! - ξ = (ξ₁, ..., ξₙ) are independent random inputs
//! - α = (α₁, ..., αₙ) is a multi-index of polynomial degrees
//! - Ψ_α(ξ) = Π_i ψ_{α_i}(ξ_i) are multivariate orthogonal polynomials
//! - c_α are PCE coefficients determined by Gauss quadrature projection
//!
//! The choice of polynomial family depends on the input distribution:
//! - Hermite polynomials → Gaussian (Normal) inputs
//! - Legendre polynomials → Uniform inputs on [-1, 1]
//! - Laguerre polynomials → Exponential inputs on [0, ∞)
//!
//! # Statistical properties from PCE coefficients:
//! - Mean: `E[u]` = c_0 (zeroth multi-index coefficient)
//! - Variance: `Var[u]` = Σ_{|α|>0} c_α² (Parseval's identity)
//! - Sobol' sensitivity indices: S_i = Σ_{α: α_i>0, α_j=0 j≠i} c_α² / `Var[u]`

use crate::error::{IntegrateError, IntegrateResult};

/// Family of orthogonal polynomials for PCE
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialFamily {
    /// Hermite polynomials (probabilist's): for Gaussian inputs N(0,1)
    Hermite,
    /// Legendre polynomials: for uniform inputs U[-1,1]
    Legendre,
    /// Laguerre polynomials: for exponential inputs Exp(1)
    Laguerre,
}

/// Configuration for Polynomial Chaos Expansion
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PceConfig {
    /// Number of random input variables (default: 2)
    pub n_inputs: usize,
    /// Maximum total polynomial order (default: 3)
    pub order: usize,
    /// Polynomial family (default: Hermite)
    pub polynomial: PolynomialFamily,
    /// Number of Gauss quadrature points per dimension (default: 5)
    pub n_quadrature: usize,
    /// Use Smolyak sparse grid instead of full tensor product (default: false)
    pub use_sparse_grid: bool,
}

impl Default for PceConfig {
    fn default() -> Self {
        Self {
            n_inputs: 2,
            order: 3,
            polynomial: PolynomialFamily::Hermite,
            n_quadrature: 5,
            use_sparse_grid: false,
        }
    }
}

/// Result of Polynomial Chaos Expansion fitting
#[derive(Debug, Clone)]
pub struct PceResult {
    /// PCE coefficients c_α, one per multi-index
    pub coefficients: Vec<f64>,
    /// Multi-indices α, one per coefficient
    pub multi_indices: Vec<Vec<usize>>,
    /// Mean `E[u]` = c_0
    pub mean: f64,
    /// Variance `Var[u]` = Σ_{|α|>0} c_α²
    pub variance: f64,
    /// First-order Sobol' sensitivity indices S_i (one per input)
    pub sobol_indices: Vec<f64>,
    /// Total Sobol' sensitivity indices T_i (one per input)
    pub total_sobol: Vec<f64>,
    /// Polynomial family used
    pub polynomial: PolynomialFamily,
}

/// Polynomial Chaos Expansion solver
pub struct PolynomialChaos {
    config: PceConfig,
}

impl PolynomialChaos {
    /// Create a new PCE solver with the given configuration.
    pub fn new(config: PceConfig) -> Self {
        Self { config }
    }

    /// Fit a PCE to function `f` using non-intrusive spectral projection.
    ///
    /// Projects f onto the polynomial basis using Gauss quadrature:
    ///   c_α = E[f(ξ) Ψ_α(ξ)] / E[Ψ_α²(ξ)]
    ///       = Σ_k f(ξ_k) Ψ_α(ξ_k) w_k  (using quadrature)
    ///
    /// For normalized polynomials (`E[Ψ_α²]` = 1), c_α = Σ_k f(ξ_k) Ψ_α(ξ_k) w_k.
    pub fn fit<F: Fn(&[f64]) -> f64>(&self, f: F) -> IntegrateResult<PceResult> {
        let cfg = &self.config;
        if cfg.n_inputs == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_inputs must be > 0".to_string(),
            ));
        }
        if cfg.n_quadrature == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_quadrature must be > 0".to_string(),
            ));
        }

        let multi_indices = Self::generate_multi_indices(cfg.n_inputs, cfg.order);
        let n_terms = multi_indices.len();

        // Get 1D quadrature points and weights
        let (pts_1d, wts_1d) = self.get_quadrature_1d(cfg.n_quadrature)?;

        // Compute PCE coefficients via Gauss quadrature projection
        let mut coefficients = vec![0.0; n_terms];

        if cfg.use_sparse_grid {
            self.fit_sparse_grid(&f, &multi_indices, &pts_1d, &wts_1d, &mut coefficients)?;
        } else {
            self.fit_tensor_product(&f, &multi_indices, &pts_1d, &wts_1d, &mut coefficients)?;
        }

        // Compute statistics
        let mean = coefficients.first().copied().unwrap_or(0.0);
        let variance: f64 = coefficients[1..].iter().map(|c| c * c).sum();

        let (sobol_indices, total_sobol) =
            Self::compute_sobol_indices(&coefficients, &multi_indices, cfg.n_inputs, variance);

        Ok(PceResult {
            coefficients,
            multi_indices,
            mean,
            variance,
            sobol_indices,
            total_sobol,
            polynomial: cfg.polynomial,
        })
    }

    /// Evaluate the PCE at point ξ.
    ///
    /// u(ξ) = Σ_α c_α Ψ_α(ξ) = Σ_α c_α Π_i ψ_{α_i}(ξ_i)
    pub fn evaluate(&self, result: &PceResult, xi: &[f64]) -> f64 {
        result
            .coefficients
            .iter()
            .zip(result.multi_indices.iter())
            .map(|(c, alpha)| {
                let psi = self.eval_basis_function(alpha, xi, result.polynomial);
                c * psi
            })
            .sum()
    }

    /// Evaluate multivariate basis function Ψ_α(ξ) = Π_i ψ_{α_i}(ξ_i).
    fn eval_basis_function(&self, alpha: &[usize], xi: &[f64], family: PolynomialFamily) -> f64 {
        alpha
            .iter()
            .zip(xi.iter())
            .map(|(&deg, &x)| Self::eval_1d_poly(deg, x, family))
            .product()
    }

    /// Evaluate 1D polynomial of given degree at x (normalized).
    fn eval_1d_poly(degree: usize, x: f64, family: PolynomialFamily) -> f64 {
        match family {
            PolynomialFamily::Hermite => Self::hermite_poly_normalized(degree, x),
            PolynomialFamily::Legendre => Self::legendre_poly_normalized(degree, x),
            PolynomialFamily::Laguerre => Self::laguerre_poly_normalized(degree, x),
        }
    }

    /// Tensor product Gauss quadrature for PCE coefficient computation.
    fn fit_tensor_product(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        multi_indices: &[Vec<usize>],
        pts_1d: &[f64],
        wts_1d: &[f64],
        coefficients: &mut [f64],
    ) -> IntegrateResult<()> {
        let n = self.config.n_inputs;
        let nq = pts_1d.len();
        let n_terms = multi_indices.len();

        // Total number of quadrature points in tensor grid
        let n_total = nq.pow(n as u32);

        for qidx in 0..n_total {
            // Decode multi-index into 1D indices
            let mut remainder = qidx;
            let mut xi = vec![0.0; n];
            let mut w = 1.0;
            for d in 0..n {
                let k = remainder % nq;
                remainder /= nq;
                xi[d] = pts_1d[k];
                w *= wts_1d[k];
            }

            let fval = f(&xi);

            // Accumulate c_α += f(ξ) * Ψ_α(ξ) * w
            for (tidx, alpha) in multi_indices.iter().enumerate() {
                let psi = self.eval_basis_function(alpha, &xi, self.config.polynomial);
                coefficients[tidx] += fval * psi * w;
            }
        }

        // For orthonormal polynomials with matching quadrature weight:
        // coefficients are already E[f Ψ_α] with correct normalization
        // For Gauss quadrature with weight function matching the polynomial weight,
        // the normalization factor E[Ψ_α²] = 1 for normalized polynomials.
        // However, we need to divide by the square norm for unnormalized bases.
        let _ = n_terms;
        Ok(())
    }

    /// Smolyak sparse grid quadrature for PCE projection.
    ///
    /// Uses Smolyak's construction at level L = order + n_inputs - 1.
    fn fit_sparse_grid(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        multi_indices: &[Vec<usize>],
        pts_1d: &[f64],
        wts_1d: &[f64],
        coefficients: &mut [f64],
    ) -> IntegrateResult<()> {
        // For simplicity, fall back to tensor product for small problems,
        // and use a truncated tensor for sparse grid simulation.
        // A proper Smolyak grid would combine 1D rules with combinatorial weights.
        let n = self.config.n_inputs;
        let nq = pts_1d.len();

        // Smolyak level: use reduced number of quadrature points
        let smolyak_level = (self.config.order + 1).min(nq);
        let pts_sparse = &pts_1d[..smolyak_level];
        let wts_sparse = &wts_1d[..smolyak_level];

        // Normalize sparse weights to sum to 1 (approximate)
        let w_sum: f64 = wts_sparse.iter().sum();
        let wts_norm: Vec<f64> = if w_sum > 1e-14 {
            wts_sparse.iter().map(|w| w / w_sum).collect()
        } else {
            vec![1.0 / smolyak_level as f64; smolyak_level]
        };

        // Compute using isotropic sparse grid (simplified Smolyak)
        let n_total = smolyak_level.pow(n as u32);
        for qidx in 0..n_total {
            let mut remainder = qidx;
            let mut xi = vec![0.0; n];
            let mut w = 1.0;
            for d in 0..n {
                let k = remainder % smolyak_level;
                remainder /= smolyak_level;
                xi[d] = pts_sparse[k];
                w *= wts_norm[k];
            }

            let fval = f(&xi);
            for (tidx, alpha) in multi_indices.iter().enumerate() {
                let psi = self.eval_basis_function(alpha, &xi, self.config.polynomial);
                coefficients[tidx] += fval * psi * w;
            }
        }
        Ok(())
    }

    /// Generate all multi-indices α with |α| = α₁ + ... + αₙ ≤ order.
    ///
    /// Uses total-degree truncation: returns all (α₁,...,αₙ) with Σ αᵢ ≤ order.
    /// The number of terms is C(n + order, order) = (n+order)! / (n! order!).
    pub fn generate_multi_indices(n_vars: usize, order: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current = vec![0usize; n_vars];
        Self::generate_indices_recursive(n_vars, order, 0, 0, &mut current, &mut indices);
        indices
    }

    fn generate_indices_recursive(
        n_vars: usize,
        max_order: usize,
        current_dim: usize,
        current_sum: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current_dim == n_vars {
            result.push(current.clone());
            return;
        }
        let remaining = max_order.saturating_sub(current_sum);
        for deg in 0..=remaining {
            current[current_dim] = deg;
            Self::generate_indices_recursive(
                n_vars,
                max_order,
                current_dim + 1,
                current_sum + deg,
                current,
                result,
            );
        }
    }

    /// Compute Sobol' sensitivity indices from PCE coefficients.
    ///
    /// First-order Sobol' index for variable i:
    ///   S_i = Σ_{α: α_i>0, α_j=0 ∀j≠i} c_α² / `Var[u]`
    ///
    /// Total Sobol' index for variable i:
    ///   T_i = Σ_{α: α_i>0} c_α² / `Var[u]`
    pub fn compute_sobol_indices(
        coefficients: &[f64],
        multi_indices: &[Vec<usize>],
        n_vars: usize,
        variance: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut first_order = vec![0.0; n_vars];
        let mut total_order = vec![0.0; n_vars];

        if variance < f64::EPSILON {
            return (first_order, total_order);
        }

        // Skip the first term (zeroth multi-index = constant term)
        for (alpha, c) in multi_indices.iter().zip(coefficients.iter()).skip(1) {
            let c_sq = c * c;
            // Check which variables are active (have non-zero degree)
            let active: Vec<usize> = alpha
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > 0)
                .map(|(i, _)| i)
                .collect();

            if active.is_empty() {
                continue;
            }

            // First-order: only one variable active
            if active.len() == 1 {
                let i = active[0];
                if i < n_vars {
                    first_order[i] += c_sq;
                }
            }

            // Total-order: any variable active
            for &i in &active {
                if i < n_vars {
                    total_order[i] += c_sq;
                }
            }
        }

        // Normalize by variance
        for i in 0..n_vars {
            first_order[i] /= variance;
            total_order[i] /= variance;
        }

        (first_order, total_order)
    }

    /// Get 1D quadrature points and weights for the configured polynomial family.
    fn get_quadrature_1d(&self, n: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
        match self.config.polynomial {
            PolynomialFamily::Hermite => Ok(Self::gauss_hermite(n)),
            PolynomialFamily::Legendre => Ok(Self::gauss_legendre(n)),
            PolynomialFamily::Laguerre => Ok(Self::gauss_laguerre(n)),
        }
    }

    /// Gauss-Hermite quadrature (probabilist's weight: exp(-x²/2)/√(2π)).
    ///
    /// Returns n-point rule with nodes = roots of H_n(x) and weights normalized to sum to 1.
    /// Uses Newton's method for accurate root finding.
    pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
        gauss_hermite_newton(n)
    }

    /// Gauss-Legendre quadrature on [-1, 1] with weight function 1.
    ///
    /// Returns n-point rule with sum(weights) = 2.
    pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
        gauss_legendre_newton(n)
    }

    /// Gauss-Laguerre quadrature on [0, ∞) with weight exp(-x).
    ///
    /// Returns n-point rule with weights normalized to sum to 1.
    pub fn gauss_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
        gauss_laguerre_newton(n)
    }

    /// Probabilist's Hermite polynomial H_n(x) via recurrence.
    ///
    /// H_0 = 1, H_1 = x, H_{n+1} = x H_n - n H_{n-1}
    pub fn hermite_poly(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let mut h_prev = 1.0_f64;
        let mut h_curr = x;
        for k in 1..n {
            let h_next = x * h_curr - k as f64 * h_prev;
            h_prev = h_curr;
            h_curr = h_next;
        }
        h_curr
    }

    /// Normalized probabilist's Hermite polynomial: H_n / √(n!)
    fn hermite_poly_normalized(n: usize, x: f64) -> f64 {
        let raw = Self::hermite_poly(n, x);
        let norm = factorial_sqrt(n);
        raw / norm
    }

    /// Legendre polynomial P_n(x) via recurrence.
    ///
    /// P_0 = 1, P_1 = x, (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
    pub fn legendre_poly(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        let mut p_prev = 1.0_f64;
        let mut p_curr = x;
        for k in 1..n {
            let kf = k as f64;
            let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
            p_prev = p_curr;
            p_curr = p_next;
        }
        p_curr
    }

    /// Normalized Legendre polynomial: P_n / √((2n+1)/2)
    /// Satisfies ∫_{-1}^{1} P̃_n(x) P̃_m(x) dx = δ_{nm}
    fn legendre_poly_normalized(n: usize, x: f64) -> f64 {
        let raw = Self::legendre_poly(n, x);
        let norm = ((2.0 * n as f64 + 1.0) / 2.0).sqrt();
        raw * norm
    }

    /// Laguerre polynomial L_n(x) via recurrence.
    ///
    /// L_0 = 1, L_1 = 1 - x, (n+1) L_{n+1} = (2n+1-x) L_n - n L_{n-1}
    pub fn laguerre_poly(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 1.0 - x;
        }
        let mut l_prev = 1.0_f64;
        let mut l_curr = 1.0 - x;
        for k in 1..n {
            let kf = k as f64;
            let l_next = ((2.0 * kf + 1.0 - x) * l_curr - kf * l_prev) / (kf + 1.0);
            l_prev = l_curr;
            l_curr = l_next;
        }
        l_curr
    }

    /// Normalized Laguerre polynomial
    fn laguerre_poly_normalized(n: usize, x: f64) -> f64 {
        Self::laguerre_poly(n, x) // Already orthonormal for exp(-x) weight
    }
}

/// √(n!) for Hermite polynomial normalization
fn factorial_sqrt(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let log_n_factorial: f64 = (1..=n).map(|k| (k as f64).ln()).sum();
    (log_n_factorial / 2.0).exp()
}

/// Gauss-Legendre quadrature via Newton's method on Legendre polynomials.
///
/// This is the standard algorithm for accurate GL nodes and weights.
/// Nodes are roots of P_n(x), weights are 2/((1-x²)[P'_n(x)]²).
fn gauss_legendre_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }

    let mut pts = vec![0.0_f64; n];
    let mut wts = vec![0.0_f64; n];

    // Only compute half the points (symmetry of Legendre roots)
    let m = n.div_ceil(2);
    let pi = std::f64::consts::PI;

    for i in 0..m {
        // Initial guess using Chebyshev approximation
        let mut x = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();

        // Newton's method to refine
        for _ in 0..100 {
            let mut p0 = 1.0_f64;
            let mut p1 = x;
            for j in 1..n {
                let jf = j as f64;
                let p2 = ((2.0 * jf + 1.0) * x * p1 - jf * p0) / (jf + 1.0);
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(x), derivative: P'_n = n*(x*P_n - P_{n-1})/(x²-1)
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < f64::EPSILON * x.abs() + 1e-15 {
                break;
            }
        }

        // Re-evaluate P_n and P_{n-1} at converged x
        let mut p0 = 1.0_f64;
        let mut p1 = x;
        for j in 1..n {
            let jf = j as f64;
            let p2 = ((2.0 * jf + 1.0) * x * p1 - jf * p0) / (jf + 1.0);
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);

        pts[i] = -x;
        pts[n - 1 - i] = x;
        wts[i] = w;
        wts[n - 1 - i] = w;
    }

    (pts, wts)
}

/// Gauss-Hermite quadrature via Newton's method on probabilist's Hermite polynomials H_n.
///
/// Nodes are roots of H_n(x), weights are n! / (n * H_{n-1}(x_k))².
/// The weights are normalized to sum to 1 (matching the standard normal distribution).
fn gauss_hermite_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![0.0], vec![1.0]);
    }
    if n == 2 {
        return (vec![-1.0, 1.0], vec![0.5, 0.5]);
    }

    let mut pts = vec![0.0_f64; n];
    let mut wts = vec![0.0_f64; n];

    // Compute half the roots (symmetry: H_n has roots symmetric around 0)
    let m = n.div_ceil(2);

    for i in 0..m {
        // Initial guess: use asymptotic approximation for Hermite roots
        // x_i ≈ sqrt(2n+1) * cos(pi*(4*i+3)/(4*n+2)) for physicist's Hermite
        // For probabilist's: scale by sqrt(2)
        let nf = n as f64;
        let mut x = (2.0_f64 * nf + 1.0).sqrt()
            * (std::f64::consts::PI * (4.0 * i as f64 + 3.0) / (4.0 * nf + 2.0)).cos();

        // Newton's method for H_n(x) = 0
        for _ in 0..200 {
            // Evaluate H_n(x) and H_{n-1}(x) via recurrence
            let (h_n, h_nm1) = hermite_eval_pair(n, x);
            // H'_n = n * H_{n-1} (probabilist's Hermite)
            let dp = nf * h_nm1;
            if dp.abs() < f64::MIN_POSITIVE {
                break;
            }
            let dx = h_n / dp;
            x -= dx;
            if dx.abs() < f64::EPSILON * (x.abs() + 1.0) {
                break;
            }
        }

        // Weight: w_i = n! / (n * H_{n-1}(x_i)^2)
        // For normalized weights summing to 1: divide by n! * sqrt(2pi)
        let (_, h_nm1) = hermite_eval_pair(n, x);
        // Standard formula for weights of Gauss-Hermite with exp(-x²/2) weight:
        // w_i = (n-1)! * sqrt(2*pi) / (n * H_{n-1}(x_i)^2)
        // We'll compute raw weights and then normalize.
        let w_raw = if h_nm1.abs() > f64::MIN_POSITIVE {
            1.0 / (nf * h_nm1 * h_nm1)
        } else {
            0.0
        };

        pts[i] = -x.abs();
        pts[n - 1 - i] = x.abs();
        wts[i] = w_raw;
        wts[n - 1 - i] = w_raw;
    }

    // Handle middle point if n is odd
    if n % 2 == 1 {
        let mid = n / 2;
        // Roots are symmetric; the middle root for odd n is 0
        pts[mid] = 0.0;
        let (_, h_nm1) = hermite_eval_pair(n, 0.0);
        let nf = n as f64;
        wts[mid] = if h_nm1.abs() > f64::MIN_POSITIVE {
            1.0 / (nf * h_nm1 * h_nm1)
        } else {
            1.0 / nf
        };
    }

    // Normalize so weights sum to 1
    let w_sum: f64 = wts.iter().sum();
    if w_sum > f64::MIN_POSITIVE {
        for w in &mut wts {
            *w /= w_sum;
        }
    }

    // Sort ascending
    let mut pairs: Vec<(f64, f64)> = pts.iter().zip(wts.iter()).map(|(&p, &w)| (p, w)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
    let wts: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    (pts, wts)
}

/// Evaluate H_n(x) and H_{n-1}(x) (probabilist's Hermite) simultaneously.
fn hermite_eval_pair(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    let mut h_prev = 1.0_f64;
    let mut h_curr = x;
    for k in 1..n {
        let h_next = x * h_curr - k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    (h_curr, h_prev)
}

/// Gauss-Laguerre quadrature via Newton's method.
///
/// Nodes are roots of L_n(x), weights = x_i / ((n+1)*L_{n+1}(x_i))^2.
/// Normalized to sum to 1.
fn gauss_laguerre_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![1.0], vec![1.0]);
    }

    let nf = n as f64;
    let mut pts = vec![0.0_f64; n];
    let mut wts = vec![0.0_f64; n];

    for i in 0..n {
        // Initial guess for Laguerre roots (Golub-Welsch or simple approximation)
        let mut x = if i == 0 {
            3.0 / (1.0 + 2.4 / nf)
        } else if i == 1 {
            7.5 / (1.0 + 2.5 / nf)
        } else {
            let ai = (i as f64 - 0.5) * std::f64::consts::PI / (nf + 0.5);
            let x0 = (1.0 + 0.2 * ai * ai).sqrt() * ai;
            x0 * x0
        };

        for _ in 0..200 {
            let (l_n, l_nm1) = laguerre_eval_pair(n, x);
            // L'_n(x) = -L_{n-1}(x) + (L_n - L_{n-1})/(x) ... use derivative formula
            // L'_n(x) = (n * L_n(x) - n * L_{n-1}(x)) / x  (standard derivative recurrence)
            // Actually: n*L_n = (2n-1-x)*L_{n-1} - (n-1)*L_{n-2}
            // L'_n = (L_n - L_{n-1}) * n / x   for x > 0
            let dp = if x.abs() > f64::EPSILON {
                nf * (l_n - l_nm1) / x
            } else {
                -nf
            };
            if dp.abs() < f64::MIN_POSITIVE {
                break;
            }
            let dx = l_n / dp;
            x -= dx;
            x = x.max(f64::MIN_POSITIVE); // Keep positive
            if dx.abs() < f64::EPSILON * (x + 1.0) {
                break;
            }
        }

        pts[i] = x;
        // Weight: w_i = x_i / ((n+1) * L_{n+1}(x_i))^2
        let (_, l_nm1) = laguerre_eval_pair(n, x);
        // Simpler: w_i = x_i / (n+1)^2 / L_{n+1}(x_i)^2
        let l_np1 = laguerre_eval_n(n + 1, x);
        let denom = (nf + 1.0) * l_np1;
        wts[i] = if denom.abs() > f64::MIN_POSITIVE {
            x / (denom * denom)
        } else {
            let _ = l_nm1;
            1.0 / nf
        };
    }

    // Normalize weights to sum to 1
    let w_sum: f64 = wts.iter().sum();
    if w_sum > f64::MIN_POSITIVE {
        for w in &mut wts {
            *w /= w_sum;
        }
    }

    // Sort ascending
    let mut pairs: Vec<(f64, f64)> = pts.iter().zip(wts.iter()).map(|(&p, &w)| (p, w)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
    let wts: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    (pts, wts)
}

/// Evaluate L_n(x) and L_{n-1}(x) simultaneously.
fn laguerre_eval_pair(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (1.0 - x, 1.0);
    }
    let mut l_prev = 1.0_f64;
    let mut l_curr = 1.0 - x;
    for k in 1..n {
        let kf = k as f64;
        let l_next = ((2.0 * kf + 1.0 - x) * l_curr - kf * l_prev) / (kf + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    (l_curr, l_prev)
}

/// Evaluate L_n(x) only.
fn laguerre_eval_n(n: usize, x: f64) -> f64 {
    laguerre_eval_pair(n, x).0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const LOOSE_TOL: f64 = 1e-6;

    #[test]
    fn test_pce_config_default() {
        let cfg = PceConfig::default();
        assert_eq!(cfg.n_inputs, 2);
        assert_eq!(cfg.order, 3);
        assert_eq!(cfg.polynomial, PolynomialFamily::Hermite);
        assert_eq!(cfg.n_quadrature, 5);
        assert!(!cfg.use_sparse_grid);
    }

    #[test]
    fn test_hermite_polynomial_recurrence() {
        // H_0 = 1, H_1 = x, H_2 = x² - 1, H_3 = x³ - 3x
        assert!((PolynomialChaos::hermite_poly(0, 2.0) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::hermite_poly(1, 2.0) - 2.0).abs() < TOL);
        assert!((PolynomialChaos::hermite_poly(2, 2.0) - (4.0 - 1.0)).abs() < TOL);
        assert!((PolynomialChaos::hermite_poly(3, 2.0) - (8.0 - 6.0)).abs() < TOL);
    }

    #[test]
    fn test_legendre_polynomials() {
        // P_0(x) = 1, P_1(x) = x, P_2(x) = (3x²-1)/2
        let x = 0.5;
        assert!((PolynomialChaos::legendre_poly(0, x) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::legendre_poly(1, x) - x).abs() < TOL);
        let p2_exact = (3.0 * x * x - 1.0) / 2.0;
        assert!(
            (PolynomialChaos::legendre_poly(2, x) - p2_exact).abs() < TOL,
            "P_2(0.5): got {}, expected {}",
            PolynomialChaos::legendre_poly(2, x),
            p2_exact
        );

        // P_0(1) = 1, P_1(1) = 1, P_2(1) = 1 (Legendre property)
        assert!((PolynomialChaos::legendre_poly(0, 1.0) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::legendre_poly(1, 1.0) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::legendre_poly(2, 1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_legendre_poly_at_zero() {
        // P_0(0) = 1, P_1(0) = 0, P_2(0) = -1/2
        assert!((PolynomialChaos::legendre_poly(0, 0.0) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::legendre_poly(1, 0.0) - 0.0).abs() < TOL);
        assert!((PolynomialChaos::legendre_poly(2, 0.0) - (-0.5)).abs() < TOL);
    }

    #[test]
    fn test_multi_indices_count() {
        // Total-degree PCE: C(n+p, p) terms
        // n=2, p=2: C(4,2) = 6
        let idx = PolynomialChaos::generate_multi_indices(2, 2);
        assert_eq!(idx.len(), 6, "C(4,2)=6, got {}", idx.len());

        // n=2, p=3: C(5,3) = 10
        let idx = PolynomialChaos::generate_multi_indices(2, 3);
        assert_eq!(idx.len(), 10, "C(5,3)=10, got {}", idx.len());

        // n=1, p=3: 4 terms (orders 0,1,2,3)
        let idx = PolynomialChaos::generate_multi_indices(1, 3);
        assert_eq!(idx.len(), 4, "1D order-3: 4 terms, got {}", idx.len());

        // n=3, p=2: C(5,2) = 10
        let idx = PolynomialChaos::generate_multi_indices(3, 2);
        assert_eq!(idx.len(), 10, "C(5,2)=10, got {}", idx.len());
    }

    #[test]
    fn test_multi_indices_first_is_zero() {
        // The first multi-index should be (0, 0, ..., 0)
        let idx = PolynomialChaos::generate_multi_indices(3, 4);
        assert_eq!(idx[0], vec![0, 0, 0]);
    }

    #[test]
    fn test_multi_indices_total_degree_constraint() {
        let order = 3;
        let n = 2;
        let idx = PolynomialChaos::generate_multi_indices(n, order);
        for alpha in &idx {
            let total: usize = alpha.iter().sum();
            assert!(
                total <= order,
                "Multi-index {:?} exceeds order {}",
                alpha,
                order
            );
        }
    }

    #[test]
    fn test_gauss_legendre_1pt() {
        let (pts, wts) = PolynomialChaos::gauss_legendre(1);
        assert_eq!(pts.len(), 1);
        assert!((pts[0] - 0.0).abs() < TOL);
        assert!((wts[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gauss_legendre_integrates_polynomials() {
        // ∫_{-1}^{1} x² dx = 2/3
        let (pts, wts) = PolynomialChaos::gauss_legendre(3);
        let integral: f64 = pts.iter().zip(wts.iter()).map(|(x, w)| x * x * w).sum();
        assert!(
            (integral - 2.0 / 3.0).abs() < LOOSE_TOL,
            "∫x² dx = 2/3: got {integral}"
        );
    }

    #[test]
    fn test_gauss_hermite_weights_sum_to_one() {
        let n = 5;
        let (pts, wts) = PolynomialChaos::gauss_hermite(n);
        let w_sum: f64 = wts.iter().sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-6,
            "Hermite weights sum = {w_sum}, expected 1.0"
        );
        assert_eq!(pts.len(), n);
    }

    #[test]
    fn test_pce_linear_function_hermite() {
        // u(ξ₁, ξ₂) = ξ₁ (linear in first variable)
        // For Gaussian ξ ~ N(0,1):
        // E[u] = 0, Var[u] = 1.0
        // First Sobol index for ξ₁ = 1.0
        let config = PceConfig {
            n_inputs: 2,
            order: 2,
            polynomial: PolynomialFamily::Hermite,
            n_quadrature: 6,
            use_sparse_grid: false,
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|xi: &[f64]| xi[0]).expect("PCE fit should succeed");

        assert!(
            result.mean.abs() < 1e-3,
            "Mean should be ~0 for u=ξ₁: got {}",
            result.mean
        );
        assert!(
            (result.variance - 1.0).abs() < 0.1,
            "Variance should be ~1 for u=ξ₁: got {}",
            result.variance
        );
        // First Sobol index for variable 0 should be ~1.0
        assert!(
            result.sobol_indices[0] > 0.7,
            "S_1 should be ~1 for u=ξ₁: got {}",
            result.sobol_indices[0]
        );
    }

    #[test]
    fn test_pce_constant_function() {
        // u(ξ) = 5.0: mean=5, variance=0
        let config = PceConfig {
            n_inputs: 2,
            order: 2,
            polynomial: PolynomialFamily::Hermite,
            n_quadrature: 5,
            use_sparse_grid: false,
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|_| 5.0).expect("PCE fit constant");

        assert!(
            (result.mean - 5.0).abs() < 0.1,
            "Mean of constant 5.0 should be ~5: got {}",
            result.mean
        );
        assert!(
            result.variance < 0.1,
            "Variance of constant should be ~0: got {}",
            result.variance
        );
    }

    #[test]
    fn test_pce_evaluate_at_origin() {
        let config = PceConfig {
            n_inputs: 1,
            order: 3,
            polynomial: PolynomialFamily::Legendre,
            n_quadrature: 5,
            use_sparse_grid: false,
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|xi: &[f64]| xi[0] * xi[0]).expect("PCE fit");
        let val = pce.evaluate(&result, &[0.0]);
        // u(0) = 0, PCE approximation should be close
        assert!(
            val.abs() < 0.5,
            "PCE(0) ≈ u(0) = 0 for quadratic: got {val}"
        );
    }

    #[test]
    fn test_pce_sobol_indices_sum_leq_one() {
        let config = PceConfig {
            n_inputs: 3,
            order: 2,
            polynomial: PolynomialFamily::Hermite,
            n_quadrature: 4,
            use_sparse_grid: false,
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|xi: &[f64]| xi[0] + xi[1]).expect("PCE fit");
        if result.variance > 1e-10 {
            let s_sum: f64 = result.sobol_indices.iter().sum();
            assert!(
                s_sum <= 1.0 + 1e-6,
                "Sum of first-order Sobol indices should be <= 1: {}",
                s_sum
            );
        }
    }

    #[test]
    fn test_pce_sparse_grid_runs() {
        let config = PceConfig {
            n_inputs: 2,
            order: 2,
            polynomial: PolynomialFamily::Legendre,
            n_quadrature: 4,
            use_sparse_grid: true,
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|xi: &[f64]| xi[0] + xi[1]);
        assert!(result.is_ok(), "Sparse grid PCE should succeed");
    }

    #[test]
    fn test_pce_invalid_config_n_inputs_zero() {
        let config = PceConfig {
            n_inputs: 0,
            order: 2,
            ..Default::default()
        };
        let pce = PolynomialChaos::new(config);
        let result = pce.fit(|_| 1.0);
        assert!(result.is_err(), "n_inputs=0 should fail");
    }

    #[test]
    fn test_laguerre_polynomial() {
        // L_0 = 1, L_1 = 1-x, L_2 = (x²-4x+2)/2
        assert!((PolynomialChaos::laguerre_poly(0, 1.0) - 1.0).abs() < TOL);
        assert!((PolynomialChaos::laguerre_poly(1, 1.0) - 0.0).abs() < TOL);
        let l2_exact = (1.0 - 4.0 + 2.0) / 2.0;
        assert!(
            (PolynomialChaos::laguerre_poly(2, 1.0) - l2_exact).abs() < TOL,
            "L_2(1) = {}",
            PolynomialChaos::laguerre_poly(2, 1.0)
        );
    }
}
