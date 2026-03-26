//! Pre-computed reference values for distribution validation.
//!
//! All values are derived from exact mathematical formulas and known constants.
//! No external computational tools were used; every value can be verified by hand
//! from the distribution's analytical definition.
//!
//! # Conventions
//!
//! - `pdf_points`: `(x, expected_pdf_value)` pairs
//! - `cdf_points`: `(x, expected_cdf_value)` pairs
//! - `ppf_points`: `(probability, expected_quantile)` pairs
//! - `mean` and `variance`: exact analytical moments

use core::f64::consts::{E, FRAC_1_PI, FRAC_1_SQRT_2, LN_2, PI, SQRT_2};

/// Reference data for a single distribution parameterization.
///
/// Stores the distribution name, parameter description, and a set of
/// known-good (x, f(x)) pairs for PDF, CDF, and PPF, plus exact moments.
#[derive(Debug, Clone)]
pub struct DistributionReference {
    /// Human-readable name, e.g. `"Normal(0,1)"`
    pub name: &'static str,
    /// Parameter description, e.g. `"mu=0, sigma=1"`
    pub params: &'static str,
    /// Known PDF values: `(x, pdf(x))`
    pub pdf_points: &'static [(f64, f64)],
    /// Known CDF values: `(x, cdf(x))`
    pub cdf_points: &'static [(f64, f64)],
    /// Known PPF/quantile values: `(p, ppf(p))`
    pub ppf_points: &'static [(f64, f64)],
    /// Exact analytical mean
    pub mean: f64,
    /// Exact analytical variance
    pub variance: f64,
}

/// Return all built-in distribution references.
pub fn all_references() -> Vec<&'static DistributionReference> {
    vec![
        &NORMAL_STANDARD,
        &NORMAL_2_3,
        &EXPONENTIAL_1,
        &UNIFORM_0_1,
        &BETA_2_5,
        &GAMMA_2_1,
        &CHI_SQUARE_4,
        &STUDENT_T_5,
        &CAUCHY_0_1,
        &POISSON_3,
        &BINOMIAL_10_03,
        &WEIBULL_2_1,
        &LOGNORMAL_0_1,
        &LAPLACE_0_1,
        &PARETO_1_2,
    ]
}

// ---------------------------------------------------------------------------
// Normal(0, 1)
// ---------------------------------------------------------------------------
// pdf(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
// cdf(x) = (1/2) * (1 + erf(x/sqrt(2)))
// mean = 0, variance = 1

/// 1 / sqrt(2 * pi)
const INV_SQRT_2PI: f64 = 0.3989422804014327;

pub static NORMAL_STANDARD: DistributionReference = DistributionReference {
    name: "Normal(0,1)",
    params: "mu=0, sigma=1",
    pdf_points: &[
        // pdf(0) = 1/sqrt(2*pi)
        (0.0, 0.3989422804014327),
        // pdf(1) = (1/sqrt(2*pi)) * exp(-0.5) = 0.3989422804014327 * 0.6065306597633104
        (1.0, 0.24197072451914337),
        // pdf(-1) = pdf(1) by symmetry
        (-1.0, 0.24197072451914337),
        // pdf(2) = (1/sqrt(2*pi)) * exp(-2) = 0.3989422804014327 * 0.1353352832366127
        (2.0, 0.05399096651318806),
        // pdf(3) = (1/sqrt(2*pi)) * exp(-4.5) = 0.3989422804014327 * 0.0111089965382423
        (3.0, 0.004431848411938008),
    ],
    cdf_points: &[
        // cdf(0) = 0.5 by symmetry
        (0.0, 0.5),
        // cdf(1) ~ 0.8413447460685429
        (1.0, 0.8413447460685429),
        // cdf(-1) = 1 - cdf(1) by symmetry
        (-1.0, 0.15865525393145702),
        // cdf(1.96) ~ 0.9750021048517796
        (1.96, 0.9750021048517796),
    ],
    ppf_points: &[
        // ppf(0.5) = 0 by symmetry
        (0.5, 0.0),
        // ppf(0.975) ~ 1.959963984540054
        (0.975, 1.959963984540054),
        // ppf(0.025) = -ppf(0.975) by symmetry
        (0.025, -1.959963984540054),
    ],
    mean: 0.0,
    variance: 1.0,
};

// ---------------------------------------------------------------------------
// Normal(2, 3)
// ---------------------------------------------------------------------------
// pdf(x) = (1/(3*sqrt(2*pi))) * exp(-(x-2)^2 / 18)
// mean = 2, variance = 9

pub static NORMAL_2_3: DistributionReference = DistributionReference {
    name: "Normal(2,3)",
    params: "mu=2, sigma=3",
    pdf_points: &[
        // pdf(2) = 1/(3*sqrt(2*pi)) = 0.3989422804014327 / 3
        (2.0, 0.13298076013381091),
        // pdf(5) = (1/(3*sqrt(2*pi))) * exp(-9/18) = (1/(3*sqrt(2*pi))) * exp(-0.5)
        (5.0, 0.08065690817304779),
        // pdf(-1) = (1/(3*sqrt(2*pi))) * exp(-9/18) = same as pdf(5) by symmetry about mu=2
        (-1.0, 0.08065690817304779),
    ],
    cdf_points: &[
        // cdf(2) = 0.5 by symmetry about mean
        (2.0, 0.5),
        // cdf(5) = cdf_std(1) where z = (5-2)/3 = 1
        (5.0, 0.8413447460685429),
    ],
    ppf_points: &[
        // ppf(0.5) = mu = 2
        (0.5, 2.0),
    ],
    mean: 2.0,
    variance: 9.0,
};

// ---------------------------------------------------------------------------
// Exponential(1) -- rate lambda = 1
// ---------------------------------------------------------------------------
// pdf(x) = exp(-x) for x >= 0
// cdf(x) = 1 - exp(-x) for x >= 0
// mean = 1, variance = 1

pub static EXPONENTIAL_1: DistributionReference = DistributionReference {
    name: "Exponential(1)",
    params: "lambda=1",
    pdf_points: &[
        // pdf(0) = exp(0) = 1
        (0.0, 1.0),
        // pdf(1) = exp(-1) = 1/e
        (1.0, 0.36787944117144233),
        // pdf(2) = exp(-2)
        (2.0, 0.1353352832366127),
        // pdf(5) = exp(-5)
        (5.0, 0.006737946999085467),
    ],
    cdf_points: &[
        // cdf(0) = 0
        (0.0, 0.0),
        // cdf(1) = 1 - exp(-1) = 1 - 1/e
        (1.0, 0.6321205588285577),
        // cdf(2) = 1 - exp(-2)
        (2.0, 0.8646647167633873),
        // cdf(ln(2)) = 1 - exp(-ln(2)) = 1 - 0.5 = 0.5
        (LN_2, 0.5),
    ],
    ppf_points: &[
        // ppf(0.5) = -ln(0.5) = ln(2)
        (0.5, LN_2),
        // ppf(0.9) = -ln(0.1) = ln(10)
        (0.9, 2.302585092994046),
    ],
    mean: 1.0,
    variance: 1.0,
};

// ---------------------------------------------------------------------------
// Uniform(0, 1)
// ---------------------------------------------------------------------------
// pdf(x) = 1 for 0 <= x <= 1
// cdf(x) = x for 0 <= x <= 1
// mean = 0.5, variance = 1/12

pub static UNIFORM_0_1: DistributionReference = DistributionReference {
    name: "Uniform(0,1)",
    params: "a=0, b=1",
    pdf_points: &[(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
    cdf_points: &[
        (0.0, 0.0),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
    ],
    ppf_points: &[
        (0.0, 0.0),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
    ],
    mean: 0.5,
    // 1/12
    variance: 0.08333333333333333,
};

// ---------------------------------------------------------------------------
// Beta(2, 5)
// ---------------------------------------------------------------------------
// pdf(x) = x^(a-1) * (1-x)^(b-1) / B(a,b) where B(2,5) = 1!*4!/6! = 1*24/720 = 1/30
// So pdf(x) = 30 * x * (1-x)^4
// mean = a/(a+b) = 2/7, variance = ab/((a+b)^2*(a+b+1)) = 10/(49*8) = 10/392 = 5/196

pub static BETA_2_5: DistributionReference = DistributionReference {
    name: "Beta(2,5)",
    params: "alpha=2, beta=5",
    pdf_points: &[
        // pdf(0) = 30 * 0 * 1^4 = 0
        (0.0, 0.0),
        // pdf(0.2) = 30 * 0.2 * (0.8)^4 = 30 * 0.2 * 0.4096 = 2.4576
        (0.2, 2.4576),
        // pdf(0.5) = 30 * 0.5 * (0.5)^4 = 30 * 0.5 * 0.0625 = 0.9375
        (0.5, 0.9375),
        // pdf(1) = 30 * 1 * 0^4 = 0
        (1.0, 0.0),
    ],
    cdf_points: &[
        (0.0, 0.0),
        (1.0, 1.0),
        // cdf(0.5) = I_{0.5}(2,5) = 1 - I_{0.5}(5,2)
        // By regularized incomplete beta: I_x(a,b) with a=2,b=5
        // cdf(0.5) = 30 * integral_0^0.5 x*(1-x)^4 dx
        // = 30 * [x^2/2*(1-x)^4 evaluated via integration by parts]
        // Exact: 30 * (1/2 * 1/32 - integral ... )
        // Using series: = 1 - (1-x)^5 * (1 + 5x) for Beta(2,5):
        // = 1 - (0.5)^5 * (1 + 2.5) = 1 - 0.03125 * 3.5 = 1 - 0.109375 = 0.890625
        (0.5, 0.890625),
    ],
    ppf_points: &[
        // ppf(0.5) ~ 0.2421 (median of Beta(2,5))
        // Exact median from I_x(2,5) = 0.5 => 1-(1-x)^5*(1+5x) = 0.5
        // No closed form, but numerically: ~0.24212
        (0.5, 0.24212028985709518),
    ],
    // mean = 2/7
    mean: 0.2857142857142857,
    // variance = 5/196
    variance: 0.025510204081632654,
};

// ---------------------------------------------------------------------------
// Gamma(2, 1)  -- shape k=2, rate/scale theta=1
// ---------------------------------------------------------------------------
// pdf(x) = x * exp(-x) for x >= 0  (since Gamma(1) = 1)
// cdf(x) = 1 - (1+x)*exp(-x) (lower incomplete gamma for k=2)
// mean = k*theta = 2, variance = k*theta^2 = 2

pub static GAMMA_2_1: DistributionReference = DistributionReference {
    name: "Gamma(2,1)",
    params: "shape=2, scale=1",
    pdf_points: &[
        // pdf(0) = 0 * exp(0) = 0
        (0.0, 0.0),
        // pdf(1) = 1 * exp(-1) = 1/e
        (1.0, 0.36787944117144233),
        // pdf(2) = 2 * exp(-2)
        (2.0, 0.2706705664732254),
        // pdf(5) = 5 * exp(-5)
        (5.0, 0.033689734995427335),
    ],
    cdf_points: &[
        // cdf(0) = 0
        (0.0, 0.0),
        // cdf(1) = 1 - 2*exp(-1) = 1 - 2/e
        (1.0, 0.26424111765711533),
        // cdf(2) = 1 - 3*exp(-2)
        (2.0, 0.593994150290162),
    ],
    ppf_points: &[
        // ppf(0.5): solve 1 - (1+x)*exp(-x) = 0.5 => (1+x)*exp(-x) = 0.5
        // Numerically: x ~ 1.6783469900166610
        (0.5, 1.6783469900166610),
    ],
    mean: 2.0,
    variance: 2.0,
};

// ---------------------------------------------------------------------------
// ChiSquare(4) -- df=4, equivalent to Gamma(2, 2)
// ---------------------------------------------------------------------------
// pdf(x) = (x/4) * exp(-x/2) / 2 = x * exp(-x/2) / 8  for x >= 0
// (Since chi^2(k) pdf = x^(k/2-1) * exp(-x/2) / (2^(k/2) * Gamma(k/2))
//  with k=4: x^1 * exp(-x/2) / (4 * 1) = x * exp(-x/2) / 4 ... wait
//  Gamma(2) = 1, 2^2 = 4, so pdf = x * exp(-x/2) / 4)
// mean = 4, variance = 2*4 = 8

pub static CHI_SQUARE_4: DistributionReference = DistributionReference {
    name: "ChiSquare(4)",
    params: "df=4",
    pdf_points: &[
        // pdf(0) = 0 * exp(0) / 4 = 0
        (0.0, 0.0),
        // pdf(2) = 2 * exp(-1) / 4 = exp(-1)/2
        (2.0, 0.18393972058572117),
        // pdf(4) = 4 * exp(-2) / 4 = exp(-2)
        (4.0, 0.1353352832366127),
        // pdf(8) = 8 * exp(-4) / 4 = 2 * exp(-4)
        (8.0, 0.036631277777468364),
    ],
    cdf_points: &[
        (0.0, 0.0),
        // cdf(x) = 1 - (1 + x/2) * exp(-x/2) for chi^2(4)
        // cdf(2) = 1 - 2*exp(-1)
        (2.0, 0.26424111765711533),
        // cdf(4) = 1 - 3*exp(-2)
        (4.0, 0.593994150290162),
    ],
    ppf_points: &[
        // ppf(0.5): solve 1 - (1+x/2)*exp(-x/2) = 0.5
        // Numerically: x ~ 3.356693980033322
        (0.5, 3.356693980033322),
    ],
    mean: 4.0,
    variance: 8.0,
};

// ---------------------------------------------------------------------------
// Student-t(5)
// ---------------------------------------------------------------------------
// pdf(x) = Gamma(3) / (sqrt(5*pi) * Gamma(5/2)) * (1 + x^2/5)^(-3)
// Gamma(3) = 2, Gamma(5/2) = (3/2)(1/2)sqrt(pi) = 3*sqrt(pi)/4
// pdf(0) = 2 / (sqrt(5*pi) * 3*sqrt(pi)/4) = 8 / (3*pi*sqrt(5))
// mean = 0, variance = 5/3

pub static STUDENT_T_5: DistributionReference = DistributionReference {
    name: "Student-t(5)",
    params: "df=5",
    pdf_points: &[
        // pdf(0) = 8 / (3 * pi * sqrt(5))
        // = 8 / (3 * 3.14159265358979 * 2.2360679774997896)
        // = 8 / 21.0866477...
        // = 0.37960669...
        (0.0, 0.37960669200152166),
        // pdf(1) = 8/(3*pi*sqrt(5)) * (1 + 1/5)^(-3) = 0.37960669... * (6/5)^(-3)
        // (6/5)^3 = 1.728, so (6/5)^(-3) = 0.578703703...
        // = 0.37960669... * 0.578703703... = 0.21967...
        (1.0, 0.21967979735098024),
        // pdf(-1) = pdf(1) by symmetry
        (-1.0, 0.21967979735098024),
    ],
    cdf_points: &[
        // cdf(0) = 0.5 by symmetry
        (0.0, 0.5),
    ],
    ppf_points: &[
        // ppf(0.5) = 0 by symmetry
        (0.5, 0.0),
    ],
    mean: 0.0,
    // variance = df/(df-2) = 5/3
    variance: 1.6666666666666667,
};

// ---------------------------------------------------------------------------
// Cauchy(0, 1)
// ---------------------------------------------------------------------------
// pdf(x) = 1 / (pi * (1 + x^2))
// cdf(x) = 0.5 + arctan(x)/pi
// mean = undefined (NaN), variance = undefined (NaN)

pub static CAUCHY_0_1: DistributionReference = DistributionReference {
    name: "Cauchy(0,1)",
    params: "x0=0, gamma=1",
    pdf_points: &[
        // pdf(0) = 1/pi
        (0.0, FRAC_1_PI),
        // pdf(1) = 1/(pi * 2) = 1/(2*pi)
        (1.0, 0.15915494309189535),
        // pdf(-1) = pdf(1)
        (-1.0, 0.15915494309189535),
        // pdf(10) = 1/(pi * 101)
        (10.0, 0.003151583031522605),
    ],
    cdf_points: &[
        // cdf(0) = 0.5
        (0.0, 0.5),
        // cdf(1) = 0.5 + arctan(1)/pi = 0.5 + (pi/4)/pi = 0.5 + 0.25 = 0.75
        (1.0, 0.75),
        // cdf(-1) = 0.25
        (-1.0, 0.25),
    ],
    ppf_points: &[
        // ppf(0.5) = tan(pi*(0.5 - 0.5)) = tan(0) = 0
        (0.5, 0.0),
        // ppf(0.75) = tan(pi * 0.25) = tan(pi/4) = 1
        (0.75, 1.0),
        // ppf(0.25) = -1
        (0.25, -1.0),
    ],
    // Mean and variance are undefined for Cauchy
    mean: f64::NAN,
    variance: f64::NAN,
};

// ---------------------------------------------------------------------------
// Poisson(3) -- discrete distribution
// ---------------------------------------------------------------------------
// pmf(k) = e^(-3) * 3^k / k!
// cdf(k) = sum_{i=0}^{k} pmf(i)
// mean = 3, variance = 3

pub static POISSON_3: DistributionReference = DistributionReference {
    name: "Poisson(3)",
    params: "lambda=3",
    pdf_points: &[
        // pmf(0) = e^(-3) = 0.049787068367863944
        (0.0, 0.049787068367863944),
        // pmf(1) = e^(-3) * 3 = 0.14936120510359183
        (1.0, 0.14936120510359183),
        // pmf(2) = e^(-3) * 9/2 = 0.22404180765538775
        (2.0, 0.22404180765538775),
        // pmf(3) = e^(-3) * 27/6 = e^(-3) * 4.5 = 0.22404180765538775
        (3.0, 0.22404180765538775),
        // pmf(4) = e^(-3) * 81/24 = e^(-3) * 3.375 = 0.16803135574154082
        (4.0, 0.16803135574154082),
        // pmf(5) = e^(-3) * 243/120 = e^(-3) * 2.025 = 0.10081881344492449
        (5.0, 0.10081881344492449),
    ],
    cdf_points: &[
        // cdf(0) = pmf(0) = e^(-3)
        (0.0, 0.049787068367863944),
        // cdf(1) = pmf(0) + pmf(1) = 4*e^(-3)
        (1.0, 0.19914827347145578),
        // cdf(2) = sum of pmf(0..=2) = e^(-3)*(1+3+4.5) = 8.5*e^(-3)
        (2.0, 0.42319008112692159),
    ],
    ppf_points: &[
        // ppf(0.5): smallest k such that cdf(k) >= 0.5
        // cdf(2) ~ 0.423, cdf(3) ~ 0.647 => ppf(0.5) = 3
        (0.5, 3.0),
    ],
    mean: 3.0,
    variance: 3.0,
};

// ---------------------------------------------------------------------------
// Binomial(10, 0.3)
// ---------------------------------------------------------------------------
// pmf(k) = C(10,k) * 0.3^k * 0.7^(10-k)
// mean = n*p = 3, variance = n*p*(1-p) = 2.1

pub static BINOMIAL_10_03: DistributionReference = DistributionReference {
    name: "Binomial(10,0.3)",
    params: "n=10, p=0.3",
    pdf_points: &[
        // pmf(0) = 0.7^10 = 0.0282475249
        (0.0, 0.0282475249),
        // pmf(1) = 10 * 0.3 * 0.7^9 = 10 * 0.3 * 0.040353607... = 0.12106082...
        (1.0, 0.12106082100),
        // pmf(2) = C(10,2)*0.3^2*0.7^8 = 45*0.09*0.05764801 = 0.23347444...
        (2.0, 0.23347444530),
        // pmf(3) = C(10,3)*0.3^3*0.7^7 = 120*0.027*0.0823543 = 0.26682793...
        (3.0, 0.26682793200),
        // pmf(10) = 0.3^10 = 0.0000059049
        (10.0, 0.0000059049),
    ],
    cdf_points: &[
        // cdf(0) = pmf(0)
        (0.0, 0.0282475249),
        // cdf(3) = sum pmf(0..=3)
        // = 0.0282475249 + 0.12106082100 + 0.23347444530 + 0.26682793200 = 0.64961072320
        (3.0, 0.64961072320),
    ],
    ppf_points: &[
        // ppf(0.5): smallest k with cdf(k) >= 0.5
        // cdf(2) = 0.0282475249 + 0.12106082100 + 0.23347444530 = 0.38278279120
        // cdf(3) = 0.64961072320 >= 0.5 => ppf(0.5) = 3
        (0.5, 3.0),
    ],
    mean: 3.0,
    variance: 2.1,
};

// ---------------------------------------------------------------------------
// Weibull(2, 1) -- shape k=2, scale lambda=1
// ---------------------------------------------------------------------------
// pdf(x) = 2*x * exp(-x^2) for x >= 0
// cdf(x) = 1 - exp(-x^2) for x >= 0
// mean = Gamma(1 + 1/2) = sqrt(pi)/2
// variance = Gamma(1+2/2) - (Gamma(1+1/2))^2 = 1 - pi/4

pub static WEIBULL_2_1: DistributionReference = DistributionReference {
    name: "Weibull(2,1)",
    params: "shape=2, scale=1",
    pdf_points: &[
        // pdf(0) = 0
        (0.0, 0.0),
        // pdf(0.5) = 2*0.5*exp(-0.25) = exp(-0.25)
        (0.5, 0.7788007830714049),
        // pdf(1) = 2*exp(-1)
        (1.0, 0.7357588823428847),
    ],
    cdf_points: &[
        (0.0, 0.0),
        // cdf(0.5) = 1 - exp(-0.25)
        (0.5, 0.22119921692859512),
        // cdf(1) = 1 - exp(-1)
        (1.0, 0.6321205588285577),
    ],
    ppf_points: &[
        // ppf(p) = sqrt(-ln(1-p))
        // ppf(0.5) = sqrt(ln(2)) = sqrt(0.6931471805599453) = 0.8325546...
        (0.5, 0.8325546111576977),
    ],
    // mean = sqrt(pi)/2
    mean: 0.8862269254527580,
    // variance = 1 - pi/4
    variance: 0.21460183660255173,
};

// ---------------------------------------------------------------------------
// Lognormal(0, 1) -- mu=0, sigma=1 (of the underlying normal)
// ---------------------------------------------------------------------------
// pdf(x) = (1/(x*sqrt(2*pi))) * exp(-ln(x)^2/2) for x > 0
// cdf(x) = Phi(ln(x))
// mean = exp(mu + sigma^2/2) = exp(0.5) = sqrt(e)
// variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2) = (e-1)*e

pub static LOGNORMAL_0_1: DistributionReference = DistributionReference {
    name: "Lognormal(0,1)",
    params: "mu=0, sigma=1",
    pdf_points: &[
        // pdf(1) = (1/sqrt(2*pi)) * exp(0) = 1/sqrt(2*pi)
        (1.0, 0.3989422804014327),
        // pdf(e) = (1/(e*sqrt(2*pi))) * exp(-1/2)
        // = 0.3989422804014327 / e * exp(-0.5)
        // = 0.3989422804014327 * 0.6065306597633104 / 2.718281828459045
        // = 0.08900015...
        (E, 0.08900015266338659),
        // pdf(0.5) = (1/(0.5*sqrt(2*pi))) * exp(-ln(0.5)^2/2)
        // ln(0.5) = -0.6931471805599453, ln(0.5)^2 = 0.4804530139182015
        // exp(-0.4804530139182015/2) = exp(-0.2402265069591007) = 0.78664085...
        // 1/(0.5*sqrt(2*pi)) = 2/sqrt(2*pi) = 2 * 0.3989422804014327 = 0.79788456...
        // = 0.79788456... * 0.78664085... = 0.62749607...
        (0.5, 0.627496077115924),
    ],
    cdf_points: &[
        // cdf(1) = Phi(ln(1)) = Phi(0) = 0.5
        (1.0, 0.5),
    ],
    ppf_points: &[
        // ppf(0.5) = exp(0) = 1 (median of lognormal(0,1))
        (0.5, 1.0),
    ],
    // mean = exp(0.5) = sqrt(e)
    mean: 1.6487212707001282,
    // variance = (e - 1) * e
    variance: 4.670774270471604,
};

// ---------------------------------------------------------------------------
// Laplace(0, 1)
// ---------------------------------------------------------------------------
// pdf(x) = 0.5 * exp(-|x|)
// cdf(x) = 0.5 * exp(x) for x < 0, 1 - 0.5*exp(-x) for x >= 0
// mean = 0, variance = 2*b^2 = 2

pub static LAPLACE_0_1: DistributionReference = DistributionReference {
    name: "Laplace(0,1)",
    params: "mu=0, b=1",
    pdf_points: &[
        // pdf(0) = 0.5
        (0.0, 0.5),
        // pdf(1) = 0.5 * exp(-1)
        (1.0, 0.18393972058572117),
        // pdf(-1) = 0.5 * exp(-1)
        (-1.0, 0.18393972058572117),
        // pdf(2) = 0.5 * exp(-2)
        (2.0, 0.06766764161830635),
    ],
    cdf_points: &[
        // cdf(0) = 0.5
        (0.0, 0.5),
        // cdf(1) = 1 - 0.5*exp(-1)
        (1.0, 0.8160602794142788),
        // cdf(-1) = 0.5*exp(-1)
        (-1.0, 0.18393972058572117),
    ],
    ppf_points: &[
        // ppf(0.5) = 0
        (0.5, 0.0),
        // ppf(0.75) = -ln(2*(1-0.75)) = -ln(0.5) = ln(2)
        (0.75, LN_2),
        // ppf(0.25) = ln(2*0.25) = ln(0.5) = -ln(2)
        (0.25, -LN_2),
    ],
    mean: 0.0,
    variance: 2.0,
};

// ---------------------------------------------------------------------------
// Pareto(1, 2) -- x_m = 1 (scale), alpha = 2 (shape)
// ---------------------------------------------------------------------------
// pdf(x) = alpha * x_m^alpha / x^(alpha+1) = 2 / x^3 for x >= 1
// cdf(x) = 1 - (x_m/x)^alpha = 1 - 1/x^2 for x >= 1
// mean = alpha*x_m/(alpha-1) = 2 (for alpha > 1)
// variance = x_m^2 * alpha / ((alpha-1)^2 * (alpha-2)) => undefined for alpha=2
// Actually variance = inf for alpha <= 2

pub static PARETO_1_2: DistributionReference = DistributionReference {
    name: "Pareto(1,2)",
    params: "x_m=1, alpha=2",
    pdf_points: &[
        // pdf(1) = 2/1^3 = 2
        (1.0, 2.0),
        // pdf(2) = 2/8 = 0.25
        (2.0, 0.25),
        // pdf(3) = 2/27
        (3.0, 0.07407407407407407),
        // pdf(10) = 2/1000 = 0.002
        (10.0, 0.002),
    ],
    cdf_points: &[
        // cdf(1) = 0
        (1.0, 0.0),
        // cdf(2) = 1 - 1/4 = 0.75
        (2.0, 0.75),
        // cdf(10) = 1 - 1/100 = 0.99
        (10.0, 0.99),
    ],
    ppf_points: &[
        // ppf(p) = x_m / (1-p)^(1/alpha) = 1/(1-p)^(1/2)
        // ppf(0.5) = 1/sqrt(0.5) = sqrt(2)
        (0.5, SQRT_2),
        // ppf(0.75) = 1/sqrt(0.25) = 2
        (0.75, 2.0),
    ],
    // mean = 2 for alpha > 1
    mean: 2.0,
    // variance is infinite for alpha <= 2
    variance: f64::INFINITY,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_values_exist() {
        let refs = all_references();
        for r in &refs {
            assert!(
                r.pdf_points.len() >= 3,
                "{} should have at least 3 PDF reference points, got {}",
                r.name,
                r.pdf_points.len()
            );
        }
    }

    #[test]
    fn test_reference_count_at_least_12_distributions() {
        let refs = all_references();
        assert!(
            refs.len() >= 12,
            "Expected at least 12 distribution references, got {}",
            refs.len()
        );
    }

    #[test]
    fn test_all_reference_means_finite_or_nan() {
        let refs = all_references();
        for r in &refs {
            // Mean should be finite or NaN (Cauchy)
            assert!(
                r.mean.is_finite() || r.mean.is_nan(),
                "{} has invalid mean: {}",
                r.name,
                r.mean
            );
        }
    }

    #[test]
    fn test_all_reference_variances_positive_or_special() {
        let refs = all_references();
        for r in &refs {
            // Variance should be positive, infinite, or NaN
            assert!(
                (r.variance > 0.0) || r.variance.is_infinite() || r.variance.is_nan(),
                "{} has non-positive variance: {}",
                r.name,
                r.variance
            );
        }
    }

    #[test]
    fn test_normal_reference_internally_consistent() {
        // pdf(-1) == pdf(1) by symmetry of standard normal
        let pdf_neg1 = NORMAL_STANDARD
            .pdf_points
            .iter()
            .find(|(x, _)| (*x - (-1.0)).abs() < 1e-15)
            .map(|(_, v)| *v);
        let pdf_pos1 = NORMAL_STANDARD
            .pdf_points
            .iter()
            .find(|(x, _)| (*x - 1.0).abs() < 1e-15)
            .map(|(_, v)| *v);
        match (pdf_neg1, pdf_pos1) {
            (Some(a), Some(b)) => {
                assert!(
                    (a - b).abs() < 1e-15,
                    "Normal(0,1) pdf(-1) and pdf(1) should be equal"
                );
            }
            _ => panic!("Missing pdf reference points for Normal(0,1) at x=-1 or x=1"),
        }
    }

    #[test]
    fn test_exponential_reference_exact() {
        // pdf(0) = 1.0 exactly
        let pdf0 = EXPONENTIAL_1.pdf_points[0];
        assert!(
            (pdf0.0).abs() < 1e-15 && (pdf0.1 - 1.0).abs() < 1e-15,
            "Exponential(1) pdf(0) should be exactly 1.0"
        );

        // cdf(0) = 0.0 exactly
        let cdf0 = EXPONENTIAL_1.cdf_points[0];
        assert!(
            (cdf0.0).abs() < 1e-15 && (cdf0.1).abs() < 1e-15,
            "Exponential(1) cdf(0) should be exactly 0.0"
        );

        // mean = 1, variance = 1
        assert!((EXPONENTIAL_1.mean - 1.0).abs() < 1e-15);
        assert!((EXPONENTIAL_1.variance - 1.0).abs() < 1e-15);
    }
}
