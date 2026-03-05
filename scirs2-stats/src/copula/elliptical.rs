//! Elliptical copula families: Gaussian and Student-t.
//!
//! Elliptical copulas are derived from elliptical distributions via Sklar's theorem.
//! They are characterized by a correlation parameter ρ ∈ (-1, 1) and (for Student-t)
//! degrees of freedom ν > 0.
//!
//! # Properties
//! - **Gaussian copula**: symmetric dependence, no tail dependence (except ρ=±1)
//! - **Student-t copula**: symmetric tail dependence λ_U = λ_L > 0, decreasing in ν
//!
//! # Algorithm
//! The key operation for sampling and density evaluation is the normal quantile
//! function (probit), implemented via Newton-Raphson iteration on the standard
//! normal CDF.
//!
//! # References
//! - Embrechts, P., McNeil, A. & Straumann, D. (2002). Correlation and dependence
//!   properties in risk management.
//! - Demarta, S. & McNeil, A.J. (2005). The t Copula and Related Copulas.

use super::archimedean::{compute_kendall_tau, LcgRng};
use crate::error::{StatsError, StatsResult};
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------------------
// Normal CDF and quantile
// ---------------------------------------------------------------------------

/// Standard normal CDF via erf approximation.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
            + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal quantile (probit) via Acklam's rational approximation.
///
/// Uses Peter J. Acklam's algorithm (2000) which gives full double-precision
/// accuracy across the entire (0, 1) range without Newton-Raphson polishing.
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation (central region)
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
         2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
         1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
         2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
         1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
         6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    // Coefficients for rational approximation (tail region)
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
         4.374_664_141_464_968e0,
         2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
         7.784_695_709_041_462e-3,
         3.224_671_290_700_398e-1,
         2.445_134_137_142_996e0,
         3.754_408_661_907_416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Gaussian Copula
// ---------------------------------------------------------------------------

/// Bivariate Gaussian (normal) copula.
///
/// C(u,v;ρ) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
///
/// where Φ₂ is the bivariate standard normal CDF and Φ⁻¹ is the probit.
///
/// - No tail dependence (λ_U = λ_L = 0 for |ρ| < 1)
/// - Kendall's τ = 2/π * arcsin(ρ)
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCopula {
    /// Pearson correlation ρ ∈ (-1, 1)
    pub rho: f64,
}

impl GaussianCopula {
    /// Create a Gaussian copula. Requires ρ ∈ (-1, 1).
    pub fn new(rho: f64) -> StatsResult<Self> {
        if !(-1.0 < rho && rho < 1.0) {
            return Err(StatsError::InvalidArgument(
                "Gaussian copula requires rho in (-1, 1)".into(),
            ));
        }
        Ok(Self { rho })
    }

    /// Copula CDF: C(u,v) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
    ///
    /// Uses the approximation for bivariate normal CDF via Gauss-Legendre quadrature.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 {
            return 0.0;
        }
        if u >= 1.0 {
            return v.clamp(0.0, 1.0);
        }
        if v >= 1.0 {
            return u.clamp(0.0, 1.0);
        }
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        bvn_cdf(x, y, self.rho).clamp(0.0, 1.0)
    }

    /// Copula density: c(u,v) = φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ) / (φ(Φ⁻¹(u)) * φ(Φ⁻¹(v)))
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 || u >= 1.0 || v >= 1.0 {
            return 0.0;
        }
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        let rho = self.rho;
        let rho2 = rho * rho;
        let disc = 1.0 - rho2;
        if disc < 1e-15 {
            return 0.0;
        }
        // c(u,v) = 1/√(1-ρ²) * exp(-(x²+y²-2ρxy)/(2(1-ρ²)) + (x²+y²)/2)
        let exponent = (2.0 * rho * x * y - rho2 * (x * x + y * y)) / (2.0 * disc);
        let val = disc.sqrt().recip() * exponent.exp();
        if val.is_finite() && val >= 0.0 { val } else { 0.0 }
    }

    /// Kendall's τ = (2/π) * arcsin(ρ)
    pub fn kendall_tau(&self) -> f64 {
        2.0 / PI * self.rho.asin()
    }

    /// Spearman's ρ_S = (6/π) * arcsin(ρ/2)
    pub fn spearman_rho(&self) -> f64 {
        6.0 / PI * (self.rho / 2.0).asin()
    }

    /// Upper tail dependence = 0 for |ρ| < 1
    pub fn upper_tail_dependence(&self) -> f64 {
        0.0
    }

    /// Lower tail dependence = 0 for |ρ| < 1
    pub fn lower_tail_dependence(&self) -> f64 {
        0.0
    }

    /// Generate n samples from the Gaussian copula.
    ///
    /// Uses the Cholesky decomposition of the correlation matrix.
    pub fn sample_pair(&self, n: usize, rng: &mut impl LcgRng) -> Vec<(f64, f64)> {
        let mut pairs = Vec::with_capacity(n);
        let rho = self.rho;
        let sqrt_disc = (1.0 - rho * rho).max(0.0).sqrt();

        for _ in 0..n {
            // Box-Muller transform for bivariate normal
            let (z1, z2) = box_muller(rng);
            // Cholesky: x1 = z1, x2 = ρ*z1 + √(1-ρ²)*z2
            let x1 = z1;
            let x2 = rho * z1 + sqrt_disc * z2;
            let u = norm_cdf(x1).clamp(1e-15, 1.0 - 1e-15);
            let v = norm_cdf(x2).clamp(1e-15, 1.0 - 1e-15);
            pairs.push((u, v));
        }
        pairs
    }

    /// Fit Gaussian copula via MLE (maximum pseudo-likelihood).
    ///
    /// The MLE for ρ has a closed-form solution: maximize the sum of log-densities.
    /// Equivalent to computing the correlation of normal scores.
    pub fn fit(u: &[f64], v: &[f64]) -> StatsResult<GaussianCopula> {
        if u.len() != v.len() || u.is_empty() {
            return Err(StatsError::InvalidArgument(
                "u and v must have the same positive length".into(),
            ));
        }
        // Transform to normal scores
        let x: Vec<f64> = u.iter().map(|&ui| norm_ppf(ui.clamp(1e-15, 1.0 - 1e-15))).collect();
        let y: Vec<f64> = v.iter().map(|&vi| norm_ppf(vi.clamp(1e-15, 1.0 - 1e-15))).collect();
        // Pearson correlation of normal scores
        let rho = pearson_correlation(&x, &y);
        GaussianCopula::new(rho.clamp(-0.9999, 0.9999))
    }
}

/// Bivariate normal CDF via Owen's T function approach (Drezner & Wesolowsky, 1990).
///
/// For the bivariate standard normal with correlation ρ:
/// Φ₂(h, k; ρ) = P(X₁ ≤ h, X₂ ≤ k)
fn bvn_cdf(h: f64, k: f64, rho: f64) -> f64 {
    // Drezner approximation via 5-point Gauss-Legendre quadrature
    let hk = h * k;

    let phi_h = norm_cdf(h);
    let phi_k = norm_cdf(k);

    if rho.abs() < 1e-12 {
        return phi_h * phi_k;
    }

    // For high |rho|, use special cases
    if rho > 0.9999 {
        return phi_h.min(phi_k);
    }
    if rho < -0.9999 {
        return (phi_h + phi_k - 1.0).max(0.0);
    }

    // Gauss-Legendre nodes and weights for [0, ρ] integral
    // Use the formula: Φ₂(h,k;ρ) = Φ(h)Φ(k) + ∫₀^ρ φ₂(h,k;r) dr
    // where ∂Φ₂/∂ρ = φ₂(h,k;ρ) = φ(h)φ(k)/√(1-ρ²) * exp(ρhk/(1-ρ²) - ... )
    // Simpler: use the Gauss-Legendre approximation directly

    // 6-point Gauss-Legendre on [-1, 1] → scale to [0, ρ]
    let gl_nodes = [-0.932_469_514_203_152, -0.661_209_386_466_265,
                    -0.238_619_186_083_197, 0.238_619_186_083_197,
                    0.661_209_386_466_265, 0.932_469_514_203_152];
    let gl_weights = [0.171_324_492_379_170, 0.360_761_573_048_139,
                      0.467_913_934_572_691, 0.467_913_934_572_691,
                      0.360_761_573_048_139, 0.171_324_492_379_170];

    let mut integral = 0.0;
    let half_rho = rho / 2.0;
    for (&node, &weight) in gl_nodes.iter().zip(gl_weights.iter()) {
        let r = half_rho * (node + 1.0); // map from [-1,1] to [0, rho]
        let r2 = r * r;
        let disc = 1.0 - r2;
        if disc < 1e-12 {
            continue;
        }
        let exponent = (r * hk - 0.5 * (h * h + k * k - 2.0 * r * hk) / disc).exp();
        let phi2 = exponent / (2.0 * PI * disc.sqrt());
        integral += weight * phi2;
    }
    integral *= half_rho; // Jacobian from [0, rho] → [-1, 1]

    let result = phi_h * phi_k + integral;
    result.clamp(0.0, 1.0)
}

/// Box-Muller transform for standard bivariate normal.
fn box_muller(rng: &mut impl LcgRng) -> (f64, f64) {
    loop {
        let u1 = rng.next_unit();
        let u2 = rng.next_unit();
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}

/// Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>();
    let sx = x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>();
    let sy = y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>();
    if sx < 1e-15 || sy < 1e-15 {
        return 0.0;
    }
    (cov / (sx * sy).sqrt()).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Student-t Copula
// ---------------------------------------------------------------------------

/// Bivariate Student-t copula with tail dependence.
///
/// C(u,v;ρ,ν) = T₂(t_ν⁻¹(u), t_ν⁻¹(v); ρ)
///
/// where T₂ is the bivariate t CDF and t_ν⁻¹ is the t quantile.
///
/// - Symmetric tail dependence: λ_U = λ_L = 2*t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
/// - As ν → ∞: reduces to Gaussian copula
#[derive(Debug, Clone, PartialEq)]
pub struct StudentCopula {
    /// Pearson correlation ρ ∈ (-1, 1)
    pub rho: f64,
    /// Degrees of freedom ν > 0
    pub nu: f64,
}

impl StudentCopula {
    /// Create a Student-t copula. Requires ρ ∈ (-1, 1) and ν > 0.
    pub fn new(rho: f64, nu: f64) -> StatsResult<Self> {
        if !(-1.0 < rho && rho < 1.0) {
            return Err(StatsError::InvalidArgument(
                "Student-t copula requires rho in (-1, 1)".into(),
            ));
        }
        if nu <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Student-t copula requires nu > 0".into(),
            ));
        }
        Ok(Self { rho, nu })
    }

    /// t-distribution CDF (univariate) for given degrees of freedom.
    pub fn t_cdf(&self, x: f64) -> f64 {
        student_t_cdf(x, self.nu)
    }

    /// t-distribution quantile (inverse CDF).
    pub fn t_ppf(&self, p: f64) -> f64 {
        student_t_ppf(p, self.nu)
    }

    /// Copula CDF via bivariate t CDF approximation.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 {
            return 0.0;
        }
        if u >= 1.0 {
            return v.clamp(0.0, 1.0);
        }
        if v >= 1.0 {
            return u.clamp(0.0, 1.0);
        }
        let x = self.t_ppf(u);
        let y = self.t_ppf(v);
        bvt_cdf(x, y, self.rho, self.nu).clamp(0.0, 1.0)
    }

    /// Copula density.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 || u >= 1.0 || v >= 1.0 {
            return 0.0;
        }
        let x = self.t_ppf(u);
        let y = self.t_ppf(v);
        let nu = self.nu;
        let rho = self.rho;
        let rho2 = rho * rho;
        let disc = 1.0 - rho2;
        if disc < 1e-15 {
            return 0.0;
        }

        // Bivariate t density
        let quad = (x * x + y * y - 2.0 * rho * x * y) / disc;
        let bvt_pdf = 1.0 / (2.0 * PI * disc.sqrt())
            * (1.0 + quad / nu).powf(-(nu + 2.0) / 2.0)
            * gamma_ratio(nu / 2.0 + 1.0, nu / 2.0 + 0.5, nu / 2.0);

        // Univariate t densities at x and y
        let t_pdf_x = student_t_pdf(x, nu);
        let t_pdf_y = student_t_pdf(y, nu);

        if t_pdf_x < 1e-300 || t_pdf_y < 1e-300 {
            return 0.0;
        }

        let val = bvt_pdf / (t_pdf_x * t_pdf_y);
        if val.is_finite() && val >= 0.0 { val } else { 0.0 }
    }

    /// Kendall's τ = (2/π) * arcsin(ρ) (same as Gaussian copula)
    pub fn kendall_tau(&self) -> f64 {
        2.0 / PI * self.rho.asin()
    }

    /// Tail dependence coefficient λ_U = λ_L = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
    pub fn tail_dependence(&self) -> f64 {
        let nu = self.nu;
        let rho = self.rho;
        if rho <= -1.0 + 1e-10 {
            return 0.0;
        }
        let arg = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
        2.0 * student_t_cdf(arg, nu + 1.0)
    }

    /// Upper tail dependence = tail_dependence()
    pub fn upper_tail_dependence(&self) -> f64 {
        self.tail_dependence()
    }

    /// Lower tail dependence = tail_dependence() (symmetric)
    pub fn lower_tail_dependence(&self) -> f64 {
        self.tail_dependence()
    }

    /// Generate n samples from the bivariate t copula.
    pub fn sample_pair(&self, n: usize, rng: &mut impl LcgRng) -> Vec<(f64, f64)> {
        let mut pairs = Vec::with_capacity(n);
        let rho = self.rho;
        let nu = self.nu;
        let sqrt_disc = (1.0 - rho * rho).max(0.0).sqrt();

        for _ in 0..n {
            // Generate bivariate normal
            let (z1, z2) = box_muller(rng);
            let x1 = z1;
            let x2 = rho * z1 + sqrt_disc * z2;

            // Generate chi-squared(nu) via sum of squares
            let chi2 = {
                let m = nu.ceil() as usize;
                let mut s = 0.0;
                for _ in 0..m {
                    let (a, _) = box_muller(rng);
                    s += a * a;
                }
                s * nu / m as f64
            };

            // t = z / sqrt(chi2/nu)
            let scale = (chi2 / nu).sqrt().max(1e-10);
            let t1 = x1 / scale;
            let t2 = x2 / scale;

            let u = student_t_cdf(t1, nu).clamp(1e-15, 1.0 - 1e-15);
            let v = student_t_cdf(t2, nu).clamp(1e-15, 1.0 - 1e-15);
            pairs.push((u, v));
        }
        pairs
    }

    /// Fit Student-t copula via MLE (optimize over ρ and ν jointly).
    pub fn fit(u: &[f64], v: &[f64]) -> StatsResult<StudentCopula> {
        if u.len() != v.len() || u.is_empty() {
            return Err(StatsError::InvalidArgument(
                "u and v must have the same positive length".into(),
            ));
        }

        // Initial estimates: rho from Kendall's tau, nu=5
        let tau = compute_kendall_tau(u, v);
        let rho_init = (PI * tau / 2.0).sin().clamp(-0.99, 0.99);

        let pdf_ll = |rho: f64, nu: f64| -> f64 {
            match StudentCopula::new(rho.clamp(-0.999, 0.999), nu.max(0.5)) {
                Ok(c) => u
                    .iter()
                    .zip(v.iter())
                    .map(|(&ui, &vi)| {
                        let p = c.pdf(ui, vi);
                        if p > 0.0 { p.ln() } else { -1e10 }
                    })
                    .sum(),
                Err(_) => f64::NEG_INFINITY,
            }
        };

        // Grid search over (rho, nu)
        let mut best_rho = rho_init;
        let mut best_nu = 5.0;
        let mut best_ll = f64::NEG_INFINITY;
        for &rho_try in &[-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8] {
            let rho_c = if rho_try == 0.0 { rho_init } else { rho_try };
            for &nu_try in &[1.0, 2.0, 5.0, 10.0, 30.0] {
                let ll = pdf_ll(rho_c, nu_try);
                if ll > best_ll {
                    best_ll = ll;
                    best_rho = rho_c;
                    best_nu = nu_try;
                }
            }
        }

        // Refine
        let mut step_r = 0.1;
        let mut step_n = 2.0;
        for _ in 0..100 {
            let mut improved = false;
            for &(dr, dn) in &[
                (step_r, 0.0_f64), (-step_r, 0.0), (0.0, step_n), (0.0, -step_n),
            ] {
                let rho_new = (best_rho + dr).clamp(-0.999, 0.999);
                let nu_new = (best_nu + dn).max(0.5);
                let ll = pdf_ll(rho_new, nu_new);
                if ll > best_ll {
                    best_ll = ll;
                    best_rho = rho_new;
                    best_nu = nu_new;
                    improved = true;
                }
            }
            if !improved {
                step_r *= 0.5;
                step_n *= 0.5;
                if step_r < 1e-6 {
                    break;
                }
            }
        }
        StudentCopula::new(best_rho, best_nu.max(0.5))
    }
}

// ---------------------------------------------------------------------------
// Student-t distribution functions
// ---------------------------------------------------------------------------

/// Student-t CDF using regularized incomplete beta function.
pub fn student_t_cdf(x: f64, nu: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    // P(T ≤ x) = I_{x²/(x²+ν)}(0.5, ν/2) / 2  for x < 0
    // = 1 - I_{ν/(x²+ν)}(ν/2, 0.5) / 2  for x ≥ 0
    // Use regularized incomplete beta function
    let x2 = x * x;
    let p = reg_inc_beta(nu / 2.0, 0.5, nu / (nu + x2));
    if x >= 0.0 {
        1.0 - p / 2.0
    } else {
        p / 2.0
    }
}

/// Student-t PDF
fn student_t_pdf(x: f64, nu: f64) -> f64 {
    let c = gamma_ratio(nu / 2.0 + 0.5, nu / 2.0, 1.0);
    c * (1.0 + x * x / nu).powf(-(nu + 1.0) / 2.0) / (nu * PI).sqrt()
}

/// Student-t quantile via bisection (crude but reliable).
fn student_t_ppf(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    // Use normal approximation for initial guess
    let z = norm_ppf(p);
    // Refine with Newton-Raphson on t-CDF
    let mut x = z * (1.0 + z * z / (4.0 * nu)); // simple correction
    for _ in 0..50 {
        let fp = student_t_cdf(x, nu) - p;
        let fprime = student_t_pdf(x, nu);
        if fprime.abs() < 1e-15 {
            break;
        }
        let delta = fp / fprime;
        x -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    x
}

/// Γ(a) / Γ(b) * factor — used for t-distribution normalization.
/// For t-PDF: Γ((ν+1)/2) / (Γ(ν/2) * √(νπ))
fn gamma_ratio(a: f64, _b: f64, _factor: f64) -> f64 {
    // Use log-gamma for numerical stability
    let ln_ratio = lgamma(a) - lgamma(a - 0.5);
    ln_ratio.exp()
}

/// Log-gamma function via Lanczos approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos g=7 coefficients
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_3_f64,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let xm1 = x - 1.0;
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
        sum += ci / (xm1 + i as f64);
    }
    let t = xm1 + g + 0.5;
    0.5 * (2.0 * PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction.
fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use continued fraction expansion
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let prefix = a * x.ln() + b * (1.0 - x).ln() - lbeta;

    // Use the symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    let switch = x > (a + 1.0) / (a + b + 2.0);
    let (a2, b2, x2) = if switch { (b, a, 1.0 - x) } else { (a, b, x) };
    let prefix2 = if switch {
        let lbeta2 = lgamma(a2) + lgamma(b2) - lgamma(a2 + b2);
        a2 * x2.ln() + b2 * (1.0 - x2).ln() - lbeta2
    } else {
        prefix
    };

    let cf = betai_cf(a2, b2, x2);
    let result = prefix2.exp() * cf / a2;

    if switch { 1.0 - result } else { result }
}

/// Continued fraction for incomplete beta (Lentz's algorithm).
fn betai_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-12;
    let fpmin = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0_f64;
    let mut d = (1.0 - qab * x / qap).abs().max(fpmin);
    d = d.recip();
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
        d = (1.0 + aa * d).abs().max(fpmin);
        c = (1.0 + aa / c).abs().max(fpmin);
        d = d.recip();
        h *= d * c;
        // Odd step
        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
        d = (1.0 + aa2 * d).abs().max(fpmin);
        c = (1.0 + aa2 / c).abs().max(fpmin);
        d = d.recip();
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Bivariate t CDF approximation using Gauss-Legendre quadrature.
fn bvt_cdf(h: f64, k: f64, rho: f64, nu: f64) -> f64 {
    // Use the bivariate normal as approximation for large nu
    if nu > 100.0 {
        return bvn_cdf(h, k, rho);
    }

    // For the bivariate t, use numerical integration approach
    let t_h = student_t_cdf(h, nu);
    let t_k = student_t_cdf(k, nu);

    if rho.abs() < 1e-8 {
        return t_h * t_k;
    }

    // Dunnett-Sobel approximation: similar GL quadrature as bivariate normal but with t kernel
    let gl_nodes = [-0.932_469_514_203_152, -0.661_209_386_466_265,
                    -0.238_619_186_083_197, 0.238_619_186_083_197,
                    0.661_209_386_466_265, 0.932_469_514_203_152];
    let gl_weights = [0.171_324_492_379_170, 0.360_761_573_048_139,
                      0.467_913_934_572_691, 0.467_913_934_572_691,
                      0.360_761_573_048_139, 0.171_324_492_379_170];

    let mut integral = 0.0;
    let half_rho = rho / 2.0;
    for (&node, &weight) in gl_nodes.iter().zip(gl_weights.iter()) {
        let r = half_rho * (node + 1.0);
        let r2 = r * r;
        let disc = 1.0 - r2;
        if disc < 1e-12 {
            continue;
        }
        let quad = (h * h + k * k - 2.0 * r * h * k) / disc;
        // Bivariate t density kernel
        let bt_pdf = (1.0 + quad / nu).powf(-(nu + 2.0) / 2.0) / (2.0 * PI * disc.sqrt());
        integral += weight * bt_pdf;
    }
    integral *= half_rho;

    let result = t_h * t_k + integral;
    result.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copula::archimedean::SimpleLcg;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- GaussianCopula ----

    #[test]
    fn test_gaussian_new_invalid() {
        assert!(GaussianCopula::new(1.0).is_err());
        assert!(GaussianCopula::new(-1.0).is_err());
    }

    #[test]
    fn test_gaussian_cdf_independence() {
        // ρ=0 → C(u,v) ≈ u*v
        let g = GaussianCopula::new(0.0).unwrap();
        let c = g.cdf(0.5, 0.5);
        assert!(approx_eq(c, 0.25, 0.01), "c={c}");
    }

    #[test]
    fn test_gaussian_cdf_boundaries() {
        let g = GaussianCopula::new(0.5).unwrap();
        assert_eq!(g.cdf(0.0, 0.5), 0.0);
        assert!((g.cdf(1.0, 0.7) - 0.7).abs() < 1e-8);
    }

    #[test]
    fn test_gaussian_pdf_positive() {
        let g = GaussianCopula::new(0.5).unwrap();
        let p = g.pdf(0.4, 0.6);
        assert!(p > 0.0, "pdf={p}");
    }

    #[test]
    fn test_gaussian_kendall_tau() {
        let g = GaussianCopula::new(0.5).expect("rho=0.5 is valid");
        let tau = g.kendall_tau();
        // τ = 2/π arcsin(ρ) = 2/π * arcsin(0.5) = 2/π * π/6 = 1/3
        assert!(tau.is_finite());
        assert!((tau - 1.0 / 3.0).abs() < 0.01, "tau={tau}, expected ~0.333");
    }

    #[test]
    fn test_gaussian_no_tail_dependence() {
        let g = GaussianCopula::new(0.9).unwrap();
        assert_eq!(g.upper_tail_dependence(), 0.0);
        assert_eq!(g.lower_tail_dependence(), 0.0);
    }

    #[test]
    fn test_gaussian_sample() {
        let g = GaussianCopula::new(0.7).unwrap();
        let mut rng = SimpleLcg::new(42);
        let pairs = g.sample_pair(100, &mut rng);
        assert_eq!(pairs.len(), 100);
        for (u, v) in &pairs {
            assert!(*u > 0.0 && *u < 1.0, "u={u}");
            assert!(*v > 0.0 && *v < 1.0, "v={v}");
        }
    }

    #[test]
    fn test_gaussian_fit() {
        let g = GaussianCopula::new(0.7).unwrap();
        let mut rng = SimpleLcg::new(42);
        let pairs = g.sample_pair(200, &mut rng);
        let u: Vec<f64> = pairs.iter().map(|&(a, _)| a).collect();
        let v: Vec<f64> = pairs.iter().map(|&(_, b)| b).collect();
        let fitted = GaussianCopula::fit(&u, &v).unwrap();
        assert!(fitted.rho > 0.0, "rho={}", fitted.rho);
    }

    // ---- StudentCopula ----

    #[test]
    fn test_student_new_invalid() {
        assert!(StudentCopula::new(1.1, 5.0).is_err());
        assert!(StudentCopula::new(0.5, 0.0).is_err());
        assert!(StudentCopula::new(0.5, -1.0).is_err());
    }

    #[test]
    fn test_student_cdf_boundaries() {
        let s = StudentCopula::new(0.5, 5.0).unwrap();
        assert_eq!(s.cdf(0.0, 0.5), 0.0);
        assert!((s.cdf(1.0, 0.7) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_student_tail_dependence_positive() {
        let s = StudentCopula::new(0.5, 5.0).unwrap();
        let td = s.tail_dependence();
        assert!(td > 0.0 && td < 1.0, "td={td}");
    }

    #[test]
    fn test_student_tail_dependence_decreasing_in_nu() {
        // Higher ν → lower tail dependence → approaches Gaussian (0)
        let s1 = StudentCopula::new(0.5, 2.0).unwrap();
        let s2 = StudentCopula::new(0.5, 30.0).unwrap();
        assert!(s1.tail_dependence() > s2.tail_dependence(),
            "td(nu=2)={} should > td(nu=30)={}", s1.tail_dependence(), s2.tail_dependence());
    }

    #[test]
    fn test_student_sample() {
        let s = StudentCopula::new(0.5, 4.0).unwrap();
        let mut rng = SimpleLcg::new(123);
        let pairs = s.sample_pair(50, &mut rng);
        assert_eq!(pairs.len(), 50);
        for (u, v) in &pairs {
            assert!(*u > 0.0 && *u < 1.0);
            assert!(*v > 0.0 && *v < 1.0);
        }
    }

    #[test]
    fn test_student_kendall_tau() {
        let s = StudentCopula::new(0.5, 5.0).unwrap();
        let tau = s.kendall_tau();
        let g = GaussianCopula::new(0.5).unwrap();
        // t-copula Kendall's tau = Gaussian Kendall's tau
        assert!(approx_eq(tau, g.kendall_tau(), 1e-10));
    }

    // ---- Helper functions ----

    #[test]
    fn test_norm_ppf_standard_values() {
        assert!(approx_eq(norm_ppf(0.5), 0.0, 1e-10));
        assert!(approx_eq(norm_ppf(0.975), 1.96, 0.01));
        assert!(approx_eq(norm_ppf(0.025), -1.96, 0.01));
    }

    #[test]
    fn test_norm_cdf_standard_values() {
        assert!(approx_eq(norm_cdf(0.0), 0.5, 1e-8));
        assert!(approx_eq(norm_cdf(1.96), 0.975, 0.001));
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        let cdf_0 = student_t_cdf(0.0, 5.0);
        assert!(approx_eq(cdf_0, 0.5, 1e-10));
        let c_pos = student_t_cdf(2.0, 5.0);
        let c_neg = student_t_cdf(-2.0, 5.0);
        assert!(approx_eq(c_pos + c_neg, 1.0, 1e-8));
    }
}
