//! Copula Models for Dependence Structures
//!
//! This module provides copula functions for modeling multivariate dependence
//! structures independently from marginal distributions:
//!
//! - **Gaussian copula**: normal dependence, no tail dependence
//! - **Student-t copula**: symmetric tail dependence
//! - **Clayton copula**: lower tail dependence (Archimedean)
//! - **Gumbel copula**: upper tail dependence (Archimedean)
//! - **Frank copula**: symmetric, no tail dependence (Archimedean)
//! - **Fitting**: maximum pseudo-likelihood estimation
//! - **Tail dependence**: upper and lower tail dependence coefficients
//!
//! # References
//!
//! - Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges.
//! - Nelsen, R.B. (2006). An Introduction to Copulas. Springer.
//! - Joe, H. (2014). Dependence Modeling with Copulas. Chapman & Hall/CRC.
//! - Genest, C. & Favre, A.-C. (2007). Everything You Always Wanted to Know
//!   About Copula Modeling but Were Afraid to Ask. J. Hydrol. Eng.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Copula trait
// ---------------------------------------------------------------------------

/// A bivariate copula C(u, v) where u, v are in [0, 1].
pub trait Copula: std::fmt::Debug {
    /// Evaluate the copula CDF: C(u, v)
    fn cdf(&self, u: f64, v: f64) -> f64;

    /// Evaluate the copula density: c(u, v) = d^2 C / du dv
    fn pdf(&self, u: f64, v: f64) -> f64;

    /// Log-density: ln(c(u, v))
    fn log_pdf(&self, u: f64, v: f64) -> f64 {
        let d = self.pdf(u, v);
        if d > 0.0 {
            d.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Conditional CDF: C(v | u) = dC/du
    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64;

    /// Conditional CDF: C(u | v) = dC/dv
    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64;

    /// Upper tail dependence coefficient lambda_U
    fn upper_tail_dependence(&self) -> f64;

    /// Lower tail dependence coefficient lambda_L
    fn lower_tail_dependence(&self) -> f64;

    /// Kendall's tau (rank correlation) implied by the copula
    fn kendalls_tau(&self) -> f64;

    /// Name of the copula family
    fn family_name(&self) -> &str;

    /// Number of parameters
    fn n_params(&self) -> usize;

    /// Get parameter vector
    fn params(&self) -> Vec<f64>;
}

// ---------------------------------------------------------------------------
// Helper: standard normal CDF and inverse
// ---------------------------------------------------------------------------

/// Standard normal CDF (Phi)
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun erf approximation
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal quantile function (inverse CDF) via rational approximation
/// Beasley-Springer-Moro algorithm
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    // Peter Acklam's algorithm (accurate to ~1.15e-9)
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Standard normal PDF
fn norm_pdf(x: f64) -> f64 {
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();
    inv_sqrt_2pi * (-0.5 * x * x).exp()
}

/// Student-t CDF approximation (for small integer or real df)
fn student_t_cdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }
    // Use incomplete beta function relation
    // P(T <= x) = 1 - 0.5 * I(df/(df+x^2), df/2, 1/2) for x >= 0
    let t = df / (df + x * x);
    let ib = regularized_incomplete_beta(t, df / 2.0, 0.5);
    if x >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use symmetry if needed
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    // Lentz's continued fraction
    let lnbeta = ln_beta(a, b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lnbeta).exp() / a;
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    f = d;
    for m in 1..200 {
        let mf = m as f64;
        // Even step
        let num_even = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num_even * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_even / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        // Odd step
        let num_odd = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num_odd / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-12 {
            break;
        }
    }
    front * f
}

/// ln(Beta(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling/Lanczos approximation of ln(Gamma(x))
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation (g=7, n=9)
    let coef = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let g = 7.0;
    let xx = x - 1.0;
    let mut sum = coef[0];
    for i in 1..9 {
        sum += coef[i] / (xx + i as f64);
    }
    let t = xx + g + 0.5;
    0.5 * (2.0 * PI).ln() + (xx + 0.5) * t.ln() - t + sum.ln()
}

/// Student-t quantile (inverse CDF) via Newton's method
fn student_t_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Start from normal approximation
    let mut x = norm_ppf(p);
    // Newton iterations
    for _ in 0..30 {
        let cdf = student_t_cdf(x, df);
        let pdf = student_t_pdf(x, df);
        if pdf.abs() < 1e-30 {
            break;
        }
        let delta = (cdf - p) / pdf;
        x -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    x
}

/// Student-t PDF
fn student_t_pdf(x: f64, df: f64) -> f64 {
    let half_df = df / 2.0;
    let coef = (ln_gamma(half_df + 0.5) - ln_gamma(half_df) - 0.5 * (df * PI).ln()).exp();
    coef * (1.0 + x * x / df).powf(-(df + 1.0) / 2.0)
}

// ---------------------------------------------------------------------------
// Gaussian copula
// ---------------------------------------------------------------------------

/// Gaussian (normal) copula parameterized by correlation rho in (-1, 1).
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    /// Correlation parameter
    pub rho: f64,
}

impl GaussianCopula {
    /// Create a Gaussian copula with the given correlation.
    pub fn new(rho: f64) -> StatsResult<Self> {
        if rho <= -1.0 || rho >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "rho must be in (-1, 1), got {}",
                rho
            )));
        }
        Ok(Self { rho })
    }
}

impl Copula for GaussianCopula {
    fn cdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        // Bivariate normal CDF approximation
        bivariate_normal_cdf(x, y, self.rho)
    }

    fn pdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        let r = self.rho;
        let det = 1.0 - r * r;
        if det <= 0.0 {
            return 0.0;
        }
        let exponent = -(r * r * (x * x + y * y) - 2.0 * r * x * y) / (2.0 * det);
        exponent.exp() / det.sqrt()
    }

    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        let r = self.rho;
        let det = 1.0 - r * r;
        if det <= 0.0 {
            return v;
        }
        norm_cdf((y - r * x) / det.sqrt())
    }

    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = norm_ppf(u);
        let y = norm_ppf(v);
        let r = self.rho;
        let det = 1.0 - r * r;
        if det <= 0.0 {
            return u;
        }
        norm_cdf((x - r * y) / det.sqrt())
    }

    fn upper_tail_dependence(&self) -> f64 {
        0.0 // Gaussian copula has no tail dependence
    }

    fn lower_tail_dependence(&self) -> f64 {
        0.0
    }

    fn kendalls_tau(&self) -> f64 {
        (2.0 / PI) * self.rho.asin()
    }

    fn family_name(&self) -> &str {
        "Gaussian"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.rho]
    }
}

/// Approximate bivariate normal CDF using Drezner-Wesolowsky (1990) method
fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < 1e-10 {
        return norm_cdf(x) * norm_cdf(y);
    }
    if rho.abs() > 0.9999 {
        if rho > 0.0 {
            return norm_cdf(x.min(y));
        } else {
            return (norm_cdf(x) + norm_cdf(-y) - 1.0).max(0.0);
        }
    }
    // Gauss-Legendre quadrature approximation
    let a = -(x * x + y * y - 2.0 * rho * x * y) / (2.0 * (1.0 - rho * rho));
    let sign = if rho >= 0.0 { 1.0 } else { -1.0 };
    // Simple approximation: product + correction
    let base = norm_cdf(x) * norm_cdf(y);
    let correction = sign * norm_pdf(x) * norm_pdf(y) * rho / (1.0 - rho * rho).sqrt();
    (base + correction * (1.0 - (-a).exp()).max(0.0).min(1.0))
        .max(0.0)
        .min(1.0)
}

// ---------------------------------------------------------------------------
// Student-t copula
// ---------------------------------------------------------------------------

/// Student-t copula with correlation rho and degrees of freedom df.
#[derive(Debug, Clone)]
pub struct StudentTCopula {
    /// Correlation parameter
    pub rho: f64,
    /// Degrees of freedom
    pub df: f64,
}

impl StudentTCopula {
    /// Create a Student-t copula.
    pub fn new(rho: f64, df: f64) -> StatsResult<Self> {
        if rho <= -1.0 || rho >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "rho must be in (-1, 1), got {}",
                rho
            )));
        }
        if df <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "df must be positive, got {}",
                df
            )));
        }
        Ok(Self { rho, df })
    }
}

impl Copula for StudentTCopula {
    fn cdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = student_t_ppf(u, self.df);
        let y = student_t_ppf(v, self.df);
        // Approximate bivariate t CDF
        bivariate_normal_cdf(norm_cdf(x) * 2.0 - 1.0, norm_cdf(y) * 2.0 - 1.0, self.rho)
            .max(0.0)
            .min(1.0)
    }

    fn pdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = student_t_ppf(u, self.df);
        let y = student_t_ppf(v, self.df);
        let r = self.rho;
        let nu = self.df;
        let det = 1.0 - r * r;
        if det <= 0.0 {
            return 0.0;
        }
        // c(u,v) = f_{nu,R}(t^{-1}(u), t^{-1}(v)) / (f_nu(t^{-1}(u)) * f_nu(t^{-1}(v)))
        // Bivariate t density
        let q = (x * x - 2.0 * r * x * y + y * y) / det;
        let biv_t = (ln_gamma((nu + 2.0) / 2.0)
            - ln_gamma(nu / 2.0)
            - (nu * PI).ln()
            - 0.5 * det.ln()
            - ((nu + 2.0) / 2.0) * (1.0 + q / nu).ln())
        .exp();
        // Marginal t densities
        let fx = student_t_pdf(x, nu);
        let fy = student_t_pdf(y, nu);
        let denom = fx * fy;
        if denom < 1e-30 {
            return 0.0;
        }
        biv_t / denom
    }

    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let x = student_t_ppf(u, self.df);
        let y = student_t_ppf(v, self.df);
        let r = self.rho;
        let nu = self.df;
        let adj_df = nu + 1.0;
        let scale = ((nu + x * x) * (1.0 - r * r) / adj_df).sqrt();
        if scale < 1e-15 {
            return v;
        }
        student_t_cdf((y - r * x) / scale, adj_df)
    }

    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64 {
        // Symmetric: swap
        self.conditional_v_given_u(v, u)
    }

    fn upper_tail_dependence(&self) -> f64 {
        let nu = self.df;
        let r = self.rho;
        // lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-r)/(1+r)))
        let arg = -((nu + 1.0) * (1.0 - r) / (1.0 + r)).sqrt();
        2.0 * student_t_cdf(arg, nu + 1.0)
    }

    fn lower_tail_dependence(&self) -> f64 {
        // Symmetric for Student-t copula
        self.upper_tail_dependence()
    }

    fn kendalls_tau(&self) -> f64 {
        (2.0 / PI) * self.rho.asin()
    }

    fn family_name(&self) -> &str {
        "Student-t"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn params(&self) -> Vec<f64> {
        vec![self.rho, self.df]
    }
}

// ---------------------------------------------------------------------------
// Clayton copula
// ---------------------------------------------------------------------------

/// Clayton copula with parameter theta > 0 (or theta in [-1, 0) for negative dependence).
/// For theta > 0, it exhibits lower tail dependence.
#[derive(Debug, Clone)]
pub struct ClaytonCopula {
    /// Dependence parameter (theta > 0 for positive dependence)
    pub theta: f64,
}

impl ClaytonCopula {
    /// Create a Clayton copula with the given parameter.
    pub fn new(theta: f64) -> StatsResult<Self> {
        if theta < -1.0 || (theta.abs() < 1e-15) {
            return Err(StatsError::InvalidArgument(format!(
                "Clayton theta must be > 0 (or in [-1, 0)), got {}",
                theta
            )));
        }
        Ok(Self { theta })
    }
}

impl Copula for ClaytonCopula {
    fn cdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let val = u.powf(-th) + v.powf(-th) - 1.0;
        if val <= 0.0 {
            return 0.0;
        }
        val.powf(-1.0 / th).max(0.0).min(1.0)
    }

    fn pdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let a = u.powf(-th) + v.powf(-th) - 1.0;
        if a <= 0.0 {
            return 0.0;
        }
        (1.0 + th) * (u * v).powf(-th - 1.0) * a.powf(-2.0 - 1.0 / th)
    }

    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let a = u.powf(-th) + v.powf(-th) - 1.0;
        if a <= 0.0 {
            return v;
        }
        u.powf(-th - 1.0) * a.powf(-1.0 - 1.0 / th)
    }

    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64 {
        // Symmetric: swap arguments in formula
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let a = u.powf(-th) + v.powf(-th) - 1.0;
        if a <= 0.0 {
            return u;
        }
        v.powf(-th - 1.0) * a.powf(-1.0 - 1.0 / th)
    }

    fn upper_tail_dependence(&self) -> f64 {
        0.0 // Clayton has no upper tail dependence
    }

    fn lower_tail_dependence(&self) -> f64 {
        if self.theta > 0.0 {
            2.0_f64.powf(-1.0 / self.theta)
        } else {
            0.0
        }
    }

    fn kendalls_tau(&self) -> f64 {
        self.theta / (self.theta + 2.0)
    }

    fn family_name(&self) -> &str {
        "Clayton"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.theta]
    }
}

// ---------------------------------------------------------------------------
// Gumbel copula
// ---------------------------------------------------------------------------

/// Gumbel copula with parameter theta >= 1.
/// Exhibits upper tail dependence.
#[derive(Debug, Clone)]
pub struct GumbelCopula {
    /// Dependence parameter (theta >= 1; theta=1 gives independence)
    pub theta: f64,
}

impl GumbelCopula {
    /// Create a Gumbel copula.
    pub fn new(theta: f64) -> StatsResult<Self> {
        if theta < 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Gumbel theta must be >= 1, got {}",
                theta
            )));
        }
        Ok(Self { theta })
    }
}

impl Copula for GumbelCopula {
    fn cdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let a = (-u.ln()).powf(th) + (-v.ln()).powf(th);
        (-a.powf(1.0 / th)).exp()
    }

    fn pdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let lu = -u.ln();
        let lv = -v.ln();
        let lu_th = lu.powf(th);
        let lv_th = lv.powf(th);
        let a = lu_th + lv_th;
        let c_val = (-a.powf(1.0 / th)).exp();
        if c_val < 1e-30 {
            return 0.0;
        }
        let a_inv = a.powf(1.0 / th);
        let term1 = (lu * lv).powf(th - 1.0);
        let term2 = a.powf(2.0 / th - 2.0);
        let term3 = a_inv + th - 1.0;
        c_val * term1 * term2 * term3 / (u * v)
    }

    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let lu = -u.ln();
        let lv = -v.ln();
        let a = lu.powf(th) + lv.powf(th);
        let c_val = (-a.powf(1.0 / th)).exp();
        c_val * lu.powf(th - 1.0) * a.powf(1.0 / th - 1.0) / u
    }

    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let lu = -u.ln();
        let lv = -v.ln();
        let a = lu.powf(th) + lv.powf(th);
        let c_val = (-a.powf(1.0 / th)).exp();
        c_val * lv.powf(th - 1.0) * a.powf(1.0 / th - 1.0) / v
    }

    fn upper_tail_dependence(&self) -> f64 {
        2.0 - 2.0_f64.powf(1.0 / self.theta)
    }

    fn lower_tail_dependence(&self) -> f64 {
        0.0 // Gumbel has no lower tail dependence
    }

    fn kendalls_tau(&self) -> f64 {
        1.0 - 1.0 / self.theta
    }

    fn family_name(&self) -> &str {
        "Gumbel"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.theta]
    }
}

// ---------------------------------------------------------------------------
// Frank copula
// ---------------------------------------------------------------------------

/// Frank copula with parameter theta != 0.
/// Symmetric copula with no tail dependence.
#[derive(Debug, Clone)]
pub struct FrankCopula {
    /// Dependence parameter (theta != 0; positive = positive dependence)
    pub theta: f64,
}

impl FrankCopula {
    /// Create a Frank copula.
    pub fn new(theta: f64) -> StatsResult<Self> {
        if theta.abs() < 1e-15 {
            return Err(StatsError::InvalidArgument(
                "Frank theta must be != 0".into(),
            ));
        }
        Ok(Self { theta })
    }
}

impl Copula for FrankCopula {
    fn cdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let num = ((-th * u).exp() - 1.0) * ((-th * v).exp() - 1.0);
        let denom = (-th).exp() - 1.0;
        if denom.abs() < 1e-30 {
            return u * v;
        }
        -(1.0 + num / denom).max(1e-30).ln() / th
    }

    fn pdf(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let e_th = (-th).exp();
        let e_u = (-th * u).exp();
        let e_v = (-th * v).exp();
        let num = -th * (e_th - 1.0) * (-th * (u + v)).exp();
        let denom_inner = (e_th - 1.0) + (e_u - 1.0) * (e_v - 1.0);
        if denom_inner.abs() < 1e-30 {
            return 0.0;
        }
        let denom = denom_inner * denom_inner;
        (num / denom).abs()
    }

    fn conditional_v_given_u(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let e_u = (-th * u).exp();
        let e_v = (-th * v).exp();
        let e_th = (-th).exp();
        let num = e_u * (e_v - 1.0);
        let denom = (e_th - 1.0) + (e_u - 1.0) * (e_v - 1.0);
        if denom.abs() < 1e-30 {
            return v;
        }
        (num / denom).max(0.0).min(1.0)
    }

    fn conditional_u_given_v(&self, u: f64, v: f64) -> f64 {
        let u = u.max(1e-10).min(1.0 - 1e-10);
        let v = v.max(1e-10).min(1.0 - 1e-10);
        let th = self.theta;
        let e_u = (-th * u).exp();
        let e_v = (-th * v).exp();
        let e_th = (-th).exp();
        let num = e_v * (e_u - 1.0);
        let denom = (e_th - 1.0) + (e_u - 1.0) * (e_v - 1.0);
        if denom.abs() < 1e-30 {
            return u;
        }
        (num / denom).max(0.0).min(1.0)
    }

    fn upper_tail_dependence(&self) -> f64 {
        0.0
    }

    fn lower_tail_dependence(&self) -> f64 {
        0.0
    }

    fn kendalls_tau(&self) -> f64 {
        let th = self.theta;
        if th.abs() < 0.01 {
            return th / 9.0; // small theta approximation
        }
        // tau = 1 - 4/theta * (1 - D_1(theta)/theta)
        // where D_1 is the first Debye function
        let d1 = debye_1(th);
        1.0 - 4.0 * (1.0 - d1) / th
    }

    fn family_name(&self) -> &str {
        "Frank"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.theta]
    }
}

/// First Debye function: D_1(x) = (1/x) * integral_0^x t/(e^t - 1) dt
///
/// For negative x the integral runs from 0 to x (i.e., the signed path), which
/// gives D_1(x) > 1 for x < 0 because t/(e^t - 1) > 1 for t < 0.  Using
/// x.abs() in the step would collapse the negative case onto the positive one
/// and produce the wrong sign of Kendall's tau for the Frank copula.
fn debye_1(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    // Numerical integration with Simpson's rule along the signed path [0, x].
    // h = x / n is signed: negative when x < 0.
    let n = 100;
    let h = x / (n as f64); // signed step
    let mut sum = 0.0;
    for i in 0..=n {
        let t = (i as f64) * h; // signed t
        let f = if t.abs() < 1e-10 {
            1.0 // limit of t/(e^t - 1) as t->0
        } else {
            t / (t.exp() - 1.0)
        };
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += w * f;
    }
    // sum * h / (3 * x) = sum * (x/n) / (3*x) = sum / (3*n) which is sign-independent,
    // but we keep the formula as sum*h/(3*x) so that the sign of h and x cancel out
    // correctly (h/x = 1/n > 0 always).
    sum * h / (3.0 * x)
}

// ---------------------------------------------------------------------------
// Tail dependence coefficients
// ---------------------------------------------------------------------------

/// Compute the tail dependence coefficients for a given copula.
#[derive(Debug, Clone)]
pub struct TailDependence {
    /// Upper tail dependence coefficient lambda_U
    pub upper: f64,
    /// Lower tail dependence coefficient lambda_L
    pub lower: f64,
}

/// Compute tail dependence for any copula.
pub fn tail_dependence(copula: &dyn Copula) -> TailDependence {
    TailDependence {
        upper: copula.upper_tail_dependence(),
        lower: copula.lower_tail_dependence(),
    }
}

// ---------------------------------------------------------------------------
// Maximum pseudo-likelihood fitting
// ---------------------------------------------------------------------------

/// Result of copula fitting
#[derive(Debug, Clone)]
pub struct CopulaFitResult {
    /// Estimated parameters
    pub params: Vec<f64>,
    /// Log pseudo-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Family name
    pub family: String,
}

/// Fit a Gaussian copula to pseudo-observations using maximum pseudo-likelihood.
///
/// # Arguments
/// * `u` - First marginal pseudo-observations (should be in (0, 1))
/// * `v` - Second marginal pseudo-observations
///
/// # Example
/// ```
/// use scirs2_stats::copulas::fit_gaussian_copula;
/// use scirs2_core::ndarray::Array1;
///
/// // Generate pseudo-observations with positive dependence
/// let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]);
/// let v = Array1::from_vec(vec![0.15, 0.25, 0.35, 0.45, 0.65, 0.85, 0.88, 0.92]);
/// let result = fit_gaussian_copula(&u.view(), &v.view()).expect("fit failed");
/// assert!(result.params[0] > 0.0); // positive dependence
/// ```
pub fn fit_gaussian_copula(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> StatsResult<CopulaFitResult> {
    if u.len() != v.len() {
        return Err(StatsError::DimensionMismatch(
            "u and v must have the same length".into(),
        ));
    }
    let n = u.len();
    if n < 3 {
        return Err(StatsError::InsufficientData(
            "need at least 3 observations for copula fitting".into(),
        ));
    }
    // Grid search + refinement for rho
    let mut best_rho = 0.0;
    let mut best_ll = f64::NEG_INFINITY;
    // Coarse grid
    for i in -19..20 {
        let rho = (i as f64) * 0.05;
        if rho.abs() >= 0.999 {
            continue;
        }
        let cop = match GaussianCopula::new(rho) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let ll = pseudo_log_likelihood(u, v, &cop);
        if ll > best_ll {
            best_ll = ll;
            best_rho = rho;
        }
    }
    // Fine refinement
    let mut step = 0.025;
    for _ in 0..20 {
        let ll_plus = {
            let r = (best_rho + step).min(0.999);
            let cop = match GaussianCopula::new(r) {
                Ok(c) => c,
                Err(_) => continue,
            };
            pseudo_log_likelihood(u, v, &cop)
        };
        let ll_minus = {
            let r = (best_rho - step).max(-0.999);
            let cop = match GaussianCopula::new(r) {
                Ok(c) => c,
                Err(_) => continue,
            };
            pseudo_log_likelihood(u, v, &cop)
        };
        if ll_plus > best_ll {
            best_rho = (best_rho + step).min(0.999);
            best_ll = ll_plus;
        } else if ll_minus > best_ll {
            best_rho = (best_rho - step).max(-0.999);
            best_ll = ll_minus;
        }
        step *= 0.5;
    }

    let nf = n as f64;
    let k = 1.0;
    Ok(CopulaFitResult {
        params: vec![best_rho],
        log_likelihood: best_ll,
        aic: -2.0 * best_ll + 2.0 * k,
        bic: -2.0 * best_ll + k * nf.ln(),
        family: "Gaussian".into(),
    })
}

/// Fit a Clayton copula to pseudo-observations.
pub fn fit_clayton_copula(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
) -> StatsResult<CopulaFitResult> {
    fit_archimedean(u, v, "Clayton")
}

/// Fit a Gumbel copula to pseudo-observations.
pub fn fit_gumbel_copula(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> StatsResult<CopulaFitResult> {
    fit_archimedean(u, v, "Gumbel")
}

/// Fit a Frank copula to pseudo-observations.
pub fn fit_frank_copula(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> StatsResult<CopulaFitResult> {
    fit_archimedean(u, v, "Frank")
}

fn fit_archimedean(
    u: &ArrayView1<f64>,
    v: &ArrayView1<f64>,
    family: &str,
) -> StatsResult<CopulaFitResult> {
    if u.len() != v.len() {
        return Err(StatsError::DimensionMismatch(
            "u and v must have the same length".into(),
        ));
    }
    let n = u.len();
    if n < 3 {
        return Err(StatsError::InsufficientData(
            "need at least 3 observations".into(),
        ));
    }
    // Estimate Kendall's tau from data
    let tau = sample_kendalls_tau(u, v)?;

    // Invert tau to get theta
    let theta_init = match family {
        "Clayton" => {
            // tau = theta / (theta + 2)  =>  theta = 2*tau / (1 - tau)
            if tau <= 0.0 {
                0.1
            } else {
                (2.0 * tau / (1.0 - tau).max(0.01)).max(0.01)
            }
        }
        "Gumbel" => {
            // tau = 1 - 1/theta  =>  theta = 1/(1 - tau)
            (1.0 / (1.0 - tau).max(0.01)).max(1.0)
        }
        "Frank" => {
            // Approximate inverse for Frank's tau-theta relation
            if tau.abs() < 0.01 {
                0.1
            } else {
                tau * 9.0
            } // rough starting point
        }
        _ => 1.0,
    };

    // Grid search around the initial estimate
    let mut best_theta = theta_init;
    let mut best_ll = f64::NEG_INFINITY;
    let (lo, hi, step_count) = match family {
        "Clayton" => (0.01, 20.0, 200),
        "Gumbel" => (1.0, 20.0, 200),
        "Frank" => (-20.0, 20.0, 200),
        _ => (0.01, 20.0, 200),
    };
    for i in 0..=step_count {
        let theta = lo + (hi - lo) * (i as f64) / (step_count as f64);
        let copula: Box<dyn Copula> = match family {
            "Clayton" => match ClaytonCopula::new(theta) {
                Ok(c) => Box::new(c),
                Err(_) => continue,
            },
            "Gumbel" => match GumbelCopula::new(theta) {
                Ok(c) => Box::new(c),
                Err(_) => continue,
            },
            "Frank" => match FrankCopula::new(theta) {
                Ok(c) => Box::new(c),
                Err(_) => continue,
            },
            _ => continue,
        };
        let ll = pseudo_log_likelihood(u, v, copula.as_ref());
        if ll > best_ll && ll.is_finite() {
            best_ll = ll;
            best_theta = theta;
        }
    }

    let nf = n as f64;
    let k = 1.0;
    Ok(CopulaFitResult {
        params: vec![best_theta],
        log_likelihood: best_ll,
        aic: -2.0 * best_ll + 2.0 * k,
        bic: -2.0 * best_ll + k * nf.ln(),
        family: family.into(),
    })
}

/// Compute the pseudo-log-likelihood for a copula given observations in (0, 1).
fn pseudo_log_likelihood(u: &ArrayView1<f64>, v: &ArrayView1<f64>, copula: &dyn Copula) -> f64 {
    let n = u.len();
    let mut ll = 0.0;
    for i in 0..n {
        let ui = u[i].max(1e-6).min(1.0 - 1e-6);
        let vi = v[i].max(1e-6).min(1.0 - 1e-6);
        let log_c = copula.log_pdf(ui, vi);
        if log_c.is_finite() {
            ll += log_c;
        } else {
            ll += -50.0; // penalty for extreme values
        }
    }
    ll
}

/// Compute Kendall's tau from paired observations.
fn sample_kendalls_tau(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> StatsResult<f64> {
    let n = u.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "need at least 2 observations for Kendall's tau".into(),
        ));
    }
    let mut concordant = 0_i64;
    let mut discordant = 0_i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let du = u[i] - u[j];
            let dv = v[i] - v[j];
            let prod = du * dv;
            if prod > 0.0 {
                concordant += 1;
            } else if prod < 0.0 {
                discordant += 1;
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 {
        return Ok(0.0);
    }
    Ok((concordant - discordant) as f64 / total as f64)
}

/// Convert data to pseudo-observations (empirical CDF values).
///
/// For each variable, replaces values with their ranks divided by (n+1).
pub fn pseudo_observations(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
) -> StatsResult<(Array1<f64>, Array1<f64>)> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::DimensionMismatch(
            "x and y must have the same length".into(),
        ));
    }
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "need at least 2 observations".into(),
        ));
    }
    let u = rank_transform(x);
    let v = rank_transform(y);
    Ok((u, v))
}

fn rank_transform(x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = Array1::<f64>::zeros(n);
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = (rank as f64 + 1.0) / (n as f64 + 1.0);
    }
    ranks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gaussian_copula_cdf_bounds() {
        let cop = GaussianCopula::new(0.5).expect("should create Gaussian copula");
        let c = cop.cdf(0.5, 0.5);
        assert!(c >= 0.0 && c <= 1.0, "CDF should be in [0,1], got {}", c);
        let c00 = cop.cdf(0.0, 0.0);
        assert!(c00 >= 0.0);
        let c11 = cop.cdf(1.0, 1.0);
        assert!(c11 <= 1.0 + 1e-10);
    }

    #[test]
    fn test_gaussian_copula_pdf_positive() {
        let cop = GaussianCopula::new(0.3).expect("should create");
        let d = cop.pdf(0.5, 0.5);
        assert!(d > 0.0, "PDF should be positive at interior, got {}", d);
    }

    #[test]
    fn test_gaussian_copula_no_tail_dependence() {
        let cop = GaussianCopula::new(0.8).expect("should create");
        assert!((cop.upper_tail_dependence()).abs() < 1e-10);
        assert!((cop.lower_tail_dependence()).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_copula_kendalls_tau() {
        let cop = GaussianCopula::new(0.0).expect("should create");
        assert!((cop.kendalls_tau()).abs() < 1e-10);
        let cop2 = GaussianCopula::new(0.5).expect("should create");
        let tau = cop2.kendalls_tau();
        assert!(tau > 0.0 && tau < 1.0);
    }

    #[test]
    fn test_gaussian_copula_invalid_rho() {
        assert!(GaussianCopula::new(1.0).is_err());
        assert!(GaussianCopula::new(-1.0).is_err());
        assert!(GaussianCopula::new(1.5).is_err());
    }

    #[test]
    fn test_student_t_copula_basic() {
        let cop = StudentTCopula::new(0.5, 4.0).expect("should create");
        let c = cop.cdf(0.5, 0.5);
        assert!(c >= 0.0 && c <= 1.0);
        let d = cop.pdf(0.5, 0.5);
        assert!(d > 0.0);
    }

    #[test]
    fn test_student_t_copula_tail_dependence() {
        let cop = StudentTCopula::new(0.5, 4.0).expect("should create");
        let lambda_u = cop.upper_tail_dependence();
        let lambda_l = cop.lower_tail_dependence();
        assert!(lambda_u > 0.0, "t-copula should have upper tail dep");
        assert!(lambda_l > 0.0, "t-copula should have lower tail dep");
        // Symmetric
        assert!((lambda_u - lambda_l).abs() < 1e-10);
    }

    #[test]
    fn test_student_t_invalid() {
        assert!(StudentTCopula::new(0.5, 0.0).is_err());
        assert!(StudentTCopula::new(1.0, 4.0).is_err());
    }

    #[test]
    fn test_clayton_copula_cdf() {
        let cop = ClaytonCopula::new(2.0).expect("should create");
        let c = cop.cdf(0.5, 0.5);
        assert!(c >= 0.0 && c <= 1.0);
        // C(u, 1) should be u
        let c_u1 = cop.cdf(0.3, 0.9999);
        assert!((c_u1 - 0.3).abs() < 0.05, "C(u,1) ~ u, got {}", c_u1);
    }

    #[test]
    fn test_clayton_lower_tail_dependence() {
        let cop = ClaytonCopula::new(2.0).expect("should create");
        let lambda_l = cop.lower_tail_dependence();
        assert!(lambda_l > 0.0);
        assert!((cop.upper_tail_dependence()).abs() < 1e-10);
    }

    #[test]
    fn test_clayton_kendalls_tau() {
        let cop = ClaytonCopula::new(2.0).expect("should create");
        let tau = cop.kendalls_tau();
        assert!(
            (tau - 0.5).abs() < 1e-10,
            "tau = theta/(theta+2) = 2/4 = 0.5, got {}",
            tau
        );
    }

    #[test]
    fn test_clayton_invalid() {
        assert!(ClaytonCopula::new(0.0).is_err());
    }

    #[test]
    fn test_gumbel_copula_cdf() {
        let cop = GumbelCopula::new(2.0).expect("should create");
        let c = cop.cdf(0.5, 0.5);
        assert!(c >= 0.0 && c <= 1.0);
        // Independence at theta=1
        let ind = GumbelCopula::new(1.0).expect("should create");
        let c_ind = ind.cdf(0.5, 0.5);
        assert!(
            (c_ind - 0.25).abs() < 0.01,
            "theta=1 => independence, got {}",
            c_ind
        );
    }

    #[test]
    fn test_gumbel_upper_tail_dependence() {
        let cop = GumbelCopula::new(2.0).expect("should create");
        let lambda_u = cop.upper_tail_dependence();
        assert!(lambda_u > 0.0);
        assert!((cop.lower_tail_dependence()).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_kendalls_tau() {
        let cop = GumbelCopula::new(2.0).expect("should create");
        let tau = cop.kendalls_tau();
        assert!(
            (tau - 0.5).abs() < 1e-10,
            "tau = 1 - 1/theta = 0.5, got {}",
            tau
        );
    }

    #[test]
    fn test_gumbel_invalid() {
        assert!(GumbelCopula::new(0.5).is_err());
    }

    #[test]
    fn test_frank_copula_cdf() {
        let cop = FrankCopula::new(5.0).expect("should create");
        let c = cop.cdf(0.5, 0.5);
        assert!(c >= 0.0 && c <= 1.0);
    }

    #[test]
    fn test_frank_copula_pdf_positive() {
        let cop = FrankCopula::new(5.0).expect("should create");
        let d = cop.pdf(0.5, 0.5);
        assert!(d > 0.0);
    }

    #[test]
    fn test_frank_no_tail_dependence() {
        let cop = FrankCopula::new(10.0).expect("should create");
        assert!((cop.upper_tail_dependence()).abs() < 1e-10);
        assert!((cop.lower_tail_dependence()).abs() < 1e-10);
    }

    #[test]
    fn test_frank_invalid() {
        assert!(FrankCopula::new(0.0).is_err());
    }

    #[test]
    fn test_frank_negative_dependence() {
        let cop = FrankCopula::new(-5.0).expect("should create");
        let tau = cop.kendalls_tau();
        assert!(tau < 0.0, "negative theta => negative tau, got {}", tau);
    }

    #[test]
    fn test_tail_dependence_helper() {
        let cop = ClaytonCopula::new(3.0).expect("should create");
        let td = tail_dependence(&cop);
        assert!(td.lower > 0.0);
        assert!(td.upper < 1e-10);
    }

    #[test]
    fn test_pseudo_observations() {
        let x = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let y = Array1::from_vec(vec![50.0, 40.0, 30.0, 20.0, 10.0]);
        let (u, v) = pseudo_observations(&x.view(), &y.view()).expect("should succeed");
        assert_eq!(u.len(), 5);
        assert_eq!(v.len(), 5);
        // Rank of smallest x (10.0) should be 1/6
        assert!((u[0] - 1.0 / 6.0).abs() < 1e-10);
        // Rank of largest x (50.0) should be 5/6
        assert!((u[4] - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_fit_gaussian_copula() {
        // Positively dependent data
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from_vec(vec![0.15, 0.25, 0.28, 0.45, 0.55, 0.62, 0.73, 0.82, 0.88]);
        let result = fit_gaussian_copula(&u.view(), &v.view());
        assert!(result.is_ok());
        let r = result.expect("fit should succeed");
        assert!(
            r.params[0] > 0.0,
            "rho should be positive, got {}",
            r.params[0]
        );
        assert!(r.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_clayton_copula() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let v = Array1::from_vec(vec![0.15, 0.18, 0.35, 0.42, 0.55, 0.58, 0.72, 0.85, 0.92]);
        let result = fit_clayton_copula(&u.view(), &v.view());
        assert!(result.is_ok());
        let r = result.expect("fit should succeed");
        assert!(r.params[0] > 0.0);
    }

    #[test]
    fn test_fit_gumbel_copula() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]);
        let v = Array1::from_vec(vec![0.15, 0.25, 0.35, 0.52, 0.68, 0.82, 0.88]);
        let result = fit_gumbel_copula(&u.view(), &v.view());
        assert!(result.is_ok());
        let r = result.expect("fit should succeed");
        assert!(r.params[0] >= 1.0);
    }

    #[test]
    fn test_fit_frank_copula() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]);
        let v = Array1::from_vec(vec![0.15, 0.25, 0.32, 0.48, 0.72, 0.78, 0.92]);
        let result = fit_frank_copula(&u.view(), &v.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_fit_insufficient_data() {
        let u = Array1::from_vec(vec![0.5]);
        let v = Array1::from_vec(vec![0.5]);
        assert!(fit_gaussian_copula(&u.view(), &v.view()).is_err());
    }

    #[test]
    fn test_fit_length_mismatch() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let v = Array1::from_vec(vec![0.1, 0.2]);
        assert!(fit_gaussian_copula(&u.view(), &v.view()).is_err());
    }

    #[test]
    fn test_conditional_gaussian() {
        let cop = GaussianCopula::new(0.5).expect("should create");
        let c = cop.conditional_v_given_u(0.5, 0.5);
        assert!(
            c >= 0.0 && c <= 1.0,
            "conditional should be in [0,1], got {}",
            c
        );
        // At u=0.5, conditional should be close to 0.5 for v=0.5
        assert!((c - 0.5).abs() < 0.15);
    }

    #[test]
    fn test_sample_kendalls_tau() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let v = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let tau = sample_kendalls_tau(&u.view(), &v.view()).expect("should succeed");
        assert!(
            (tau - 1.0).abs() < 1e-10,
            "perfect concordance => tau=1, got {}",
            tau
        );
    }

    #[test]
    fn test_sample_kendalls_tau_negative() {
        let u = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let v = Array1::from_vec(vec![0.5, 0.4, 0.3, 0.2, 0.1]);
        let tau = sample_kendalls_tau(&u.view(), &v.view()).expect("should succeed");
        assert!(
            (tau - (-1.0)).abs() < 1e-10,
            "perfect discordance => tau=-1, got {}",
            tau
        );
    }

    #[test]
    fn test_norm_ppf_roundtrip() {
        for &p in &[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] {
            let x = norm_ppf(p);
            let q = norm_cdf(x);
            assert!(
                (q - p).abs() < 1e-6,
                "roundtrip failed for p={}: got {}",
                p,
                q
            );
        }
    }

    #[test]
    fn test_copula_families_comprehensive() {
        // Test all families with moderate parameters
        let families: Vec<Box<dyn Copula>> = vec![
            Box::new(GaussianCopula::new(0.5).expect("create")),
            Box::new(StudentTCopula::new(0.5, 5.0).expect("create")),
            Box::new(ClaytonCopula::new(2.0).expect("create")),
            Box::new(GumbelCopula::new(2.0).expect("create")),
            Box::new(FrankCopula::new(5.0).expect("create")),
        ];
        for cop in &families {
            let c = cop.cdf(0.5, 0.5);
            assert!(
                c >= 0.0 && c <= 1.0,
                "{} CDF out of bounds: {}",
                cop.family_name(),
                c
            );
            let d = cop.pdf(0.5, 0.5);
            assert!(d >= 0.0, "{} PDF negative: {}", cop.family_name(), d);
            let tau = cop.kendalls_tau();
            assert!(tau.is_finite(), "{} tau not finite", cop.family_name());
            let lu = cop.upper_tail_dependence();
            let ll = cop.lower_tail_dependence();
            assert!(
                lu >= 0.0 && lu <= 1.0,
                "{} upper tail dep: {}",
                cop.family_name(),
                lu
            );
            assert!(
                ll >= 0.0 && ll <= 1.0,
                "{} lower tail dep: {}",
                cop.family_name(),
                ll
            );
        }
    }
}
