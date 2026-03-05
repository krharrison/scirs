//! Dual number arithmetic for forward-mode automatic differentiation
//!
//! A *dual number* is a pair `(value, derivative)` that extends the real numbers
//! with an infinitesimal part `ε` satisfying `ε² = 0`.  Arithmetic on dual
//! numbers automatically propagates first-order derivatives via the chain rule,
//! turning ordinary Rust functions into their own derivative computation.
//!
//! # Architecture
//!
//! This module provides two related types:
//!
//! * [`Dual`] — a simple `f64` dual number suitable for scalar functions and
//!   as a building block for higher-order types.
//! * [`HyperDual`] — a dual-of-dual number `(f, f', f'', …)` supporting
//!   second-order forward-mode AD (i.e. forward-over-forward for Hessians).
//!
//! Both types implement all standard arithmetic operators, comparison traits, and
//! a subset of transcendental functions so that user code can be written against
//! plain Rust operators without any changes.
//!
//! # Example — scalar gradient
//!
//! ```rust
//! use scirs2_autograd::functional::dual::Dual;
//!
//! // f(x) = x^3 + 2*x
//! // f'(x) = 3*x^2 + 2
//! fn f(x: Dual) -> Dual {
//!     x * x * x + Dual::constant(2.0) * x
//! }
//!
//! // Seed x = 3.0 with tangent 1 to get f'(3)
//! let x = Dual::variable(3.0);
//! let y = f(x);
//! assert!((y.value - 33.0).abs() < 1e-12);     // 27 + 6 = 33
//! assert!((y.derivative - 29.0).abs() < 1e-12); // 27 + 2 = 29
//! ```
//!
//! # Example — multivariate partial derivative
//!
//! ```rust
//! use scirs2_autograd::functional::dual::Dual;
//!
//! // f(x, y) = x^2 * y + sin(y)
//! // ∂f/∂x = 2*x*y,  ∂f/∂y = x^2 + cos(y)
//! fn f(x: Dual, y: Dual) -> Dual {
//!     x * x * y + y.sin()
//! }
//!
//! // ∂f/∂x at (2, π)
//! let (x, y) = (Dual::variable(2.0), Dual::constant(std::f64::consts::PI));
//! let df_dx = f(x, y).derivative;
//! assert!((df_dx - 2.0 * 2.0 * std::f64::consts::PI).abs() < 1e-9);
//! ```

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// Dual — first-order forward-mode dual number
// ============================================================================

/// A first-order dual number `(value, derivative)`.
///
/// Arithmetic follows the rule:
///
/// `(a + b·ε) ⊕ (c + d·ε) = (a⊕c) + (d⊕-chain-rule·ε)`
///
/// where `⊕` is the corresponding real operation.
#[derive(Clone, Copy, PartialEq)]
pub struct Dual {
    /// The primal (real) value.
    pub value: f64,
    /// The tangent (derivative) component.
    pub derivative: f64,
}

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual {{ value: {}, derivative: {} }}", self.value, self.derivative)
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.derivative >= 0.0 {
            write!(f, "{} + {}ε", self.value, self.derivative)
        } else {
            write!(f, "{} - {}ε", self.value, -self.derivative)
        }
    }
}

impl Dual {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a dual number with explicit value and derivative components.
    #[inline]
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Create a *constant* dual number: `derivative = 0`.
    ///
    /// Use this for inputs that are held fixed during differentiation.
    #[inline]
    pub fn constant(value: f64) -> Self {
        Self { value, derivative: 0.0 }
    }

    /// Create an *active variable* dual number: `derivative = 1`.
    ///
    /// Use this to seed the differentiation variable in univariate problems,
    /// or to extract one partial derivative in multivariate problems.
    #[inline]
    pub fn variable(value: f64) -> Self {
        Self { value, derivative: 1.0 }
    }

    /// Create a zero dual number.
    #[inline]
    pub fn zero() -> Self {
        Self { value: 0.0, derivative: 0.0 }
    }

    /// Create a unit dual number (value=1, derivative=0).
    #[inline]
    pub fn one() -> Self {
        Self { value: 1.0, derivative: 0.0 }
    }

    // -----------------------------------------------------------------------
    // Transcendental functions
    // -----------------------------------------------------------------------

    /// `sin(a + b·ε) = sin(a) + b·cos(a)·ε`
    #[inline]
    pub fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            derivative: self.derivative * self.value.cos(),
        }
    }

    /// `cos(a + b·ε) = cos(a) − b·sin(a)·ε`
    #[inline]
    pub fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            derivative: -self.derivative * self.value.sin(),
        }
    }

    /// `tan(a + b·ε) = tan(a) + b/cos²(a)·ε`
    #[inline]
    pub fn tan(self) -> Self {
        let c = self.value.cos();
        Self {
            value: self.value.tan(),
            derivative: self.derivative / (c * c),
        }
    }

    /// `exp(a + b·ε) = exp(a) + b·exp(a)·ε`
    #[inline]
    pub fn exp(self) -> Self {
        let e = self.value.exp();
        Self {
            value: e,
            derivative: self.derivative * e,
        }
    }

    /// `ln(a + b·ε) = ln(a) + b/a·ε`  (requires `a > 0`)
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        }
    }

    /// `log2(a + b·ε) = log2(a) + b/(a·ln2)·ε`
    #[inline]
    pub fn log2(self) -> Self {
        Self {
            value: self.value.log2(),
            derivative: self.derivative / (self.value * std::f64::consts::LN_2),
        }
    }

    /// `log10(a + b·ε) = log10(a) + b/(a·ln10)·ε`
    #[inline]
    pub fn log10(self) -> Self {
        Self {
            value: self.value.log10(),
            derivative: self.derivative / (self.value * std::f64::consts::LN_10),
        }
    }

    /// `sqrt(a + b·ε) = sqrt(a) + b/(2·sqrt(a))·ε`
    #[inline]
    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        Self {
            value: s,
            derivative: self.derivative / (2.0 * s),
        }
    }

    /// `cbrt(a + b·ε) = cbrt(a) + b/(3·cbrt(a)²)·ε`
    #[inline]
    pub fn cbrt(self) -> Self {
        let c = self.value.cbrt();
        Self {
            value: c,
            derivative: self.derivative / (3.0 * c * c),
        }
    }

    /// `abs(a + b·ε) = |a| + b·sign(a)·ε`
    ///
    /// Note: not differentiable at `a = 0`; returns `derivative = 0` there.
    #[inline]
    pub fn abs(self) -> Self {
        let sign = if self.value > 0.0 {
            1.0
        } else if self.value < 0.0 {
            -1.0
        } else {
            0.0
        };
        Self {
            value: self.value.abs(),
            derivative: self.derivative * sign,
        }
    }

    /// `powi(a + b·ε, n) = a^n + n·a^(n-1)·b·ε`
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        Self {
            value: self.value.powi(n),
            derivative: self.derivative * f64::from(n) * self.value.powi(n - 1),
        }
    }

    /// `powf(a + b·ε, p) = a^p + p·a^(p-1)·b·ε`
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        Self {
            value: self.value.powf(p),
            derivative: self.derivative * p * self.value.powf(p - 1.0),
        }
    }

    /// `a^(c + d·ε) = a^c + d·a^c·ln(a)·ε` (general power, `a > 0`)
    pub fn pow_dual(self, exponent: Dual) -> Self {
        let base_pow = self.value.powf(exponent.value);
        // d/d(both) [base^exp] = base^exp * (exp * d_base/base + ln(base) * d_exp)
        let d_value = base_pow
            * (exponent.value * self.derivative / self.value
                + self.value.ln() * exponent.derivative);
        Self {
            value: base_pow,
            derivative: d_value,
        }
    }

    /// `asin(a + b·ε) = asin(a) + b/sqrt(1-a²)·ε`
    #[inline]
    pub fn asin(self) -> Self {
        Self {
            value: self.value.asin(),
            derivative: self.derivative / (1.0 - self.value * self.value).sqrt(),
        }
    }

    /// `acos(a + b·ε) = acos(a) - b/sqrt(1-a²)·ε`
    #[inline]
    pub fn acos(self) -> Self {
        Self {
            value: self.value.acos(),
            derivative: -self.derivative / (1.0 - self.value * self.value).sqrt(),
        }
    }

    /// `atan(a + b·ε) = atan(a) + b/(1+a²)·ε`
    #[inline]
    pub fn atan(self) -> Self {
        Self {
            value: self.value.atan(),
            derivative: self.derivative / (1.0 + self.value * self.value),
        }
    }

    /// `atan2(self, other)` — dual-number 2-argument arctangent.
    ///
    /// `atan2(y, x) = atan(y/x)`, derivative:
    /// `d atan2(y,x)/dt = (x·ẏ - y·ẋ) / (x² + y²)`
    pub fn atan2(self, other: Dual) -> Self {
        let denom = other.value * other.value + self.value * self.value;
        Self {
            value: self.value.atan2(other.value),
            derivative: (other.value * self.derivative - self.value * other.derivative) / denom,
        }
    }

    /// `sinh(a + b·ε) = sinh(a) + b·cosh(a)·ε`
    #[inline]
    pub fn sinh(self) -> Self {
        Self {
            value: self.value.sinh(),
            derivative: self.derivative * self.value.cosh(),
        }
    }

    /// `cosh(a + b·ε) = cosh(a) + b·sinh(a)·ε`
    #[inline]
    pub fn cosh(self) -> Self {
        Self {
            value: self.value.cosh(),
            derivative: self.derivative * self.value.sinh(),
        }
    }

    /// `tanh(a + b·ε) = tanh(a) + b·(1 - tanh²(a))·ε`
    #[inline]
    pub fn tanh(self) -> Self {
        let t = self.value.tanh();
        Self {
            value: t,
            derivative: self.derivative * (1.0 - t * t),
        }
    }

    /// `max(a, b)` — subgradient at a tie is 0.5 each.
    #[inline]
    pub fn max(self, other: Dual) -> Self {
        if self.value > other.value {
            self
        } else if other.value > self.value {
            other
        } else {
            // equal — average derivative (subgradient)
            Self {
                value: self.value,
                derivative: 0.5 * (self.derivative + other.derivative),
            }
        }
    }

    /// `min(a, b)` — subgradient at a tie is 0.5 each.
    #[inline]
    pub fn min(self, other: Dual) -> Self {
        if self.value < other.value {
            self
        } else if other.value < self.value {
            other
        } else {
            Self {
                value: self.value,
                derivative: 0.5 * (self.derivative + other.derivative),
            }
        }
    }

    /// ReLU: `max(0, x)`.
    #[inline]
    pub fn relu(self) -> Self {
        if self.value > 0.0 {
            self
        } else if self.value < 0.0 {
            Self::zero()
        } else {
            Self::zero() // subgradient 0 at kink
        }
    }

    /// Sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        let s = 1.0 / (1.0 + (-self.value).exp());
        Self {
            value: s,
            derivative: self.derivative * s * (1.0 - s),
        }
    }

    /// Softplus: `ln(1 + exp(x))`.
    pub fn softplus(self) -> Self {
        let ep = self.value.exp();
        Self {
            value: (1.0 + ep).ln(),
            derivative: self.derivative * ep / (1.0 + ep),
        }
    }

    // -----------------------------------------------------------------------
    // Predicates / comparisons (on primal value only)
    // -----------------------------------------------------------------------

    /// Return `true` if the primal value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    /// Return `true` if the primal value is infinite.
    #[inline]
    pub fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }

    /// Return `true` if the primal value is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    /// Clamp the primal value to `[min, max]` and propagate derivative
    /// (derivative is zeroed if clamped).
    #[inline]
    pub fn clamp(self, min: f64, max: f64) -> Self {
        if self.value < min {
            Self::constant(min)
        } else if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }
}

// ============================================================================
// Arithmetic operator implementations
// ============================================================================

impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { value: -self.value, derivative: -self.derivative }
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self { value: self.value + rhs, derivative: self.derivative }
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn add(self, rhs: Dual) -> Dual {
        Dual { value: self + rhs.value, derivative: rhs.derivative }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self { value: self.value - rhs, derivative: self.derivative }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn sub(self, rhs: Dual) -> Dual {
        Dual { value: self - rhs.value, derivative: -rhs.derivative }
    }
}

impl Mul for Dual {
    type Output = Self;
    /// Product rule: `(a·ε) * (b·ε) = a·b + (a·b' + a'·b)·ε`
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self { value: self.value * rhs, derivative: self.derivative * rhs }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn mul(self, rhs: Dual) -> Dual {
        Dual { value: self * rhs.value, derivative: self * rhs.derivative }
    }
}

impl Div for Dual {
    type Output = Self;
    /// Quotient rule: `(a + a'·ε) / (b + b'·ε) = a/b + (a'·b − a·b') / b²·ε`
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let b2 = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) / b2,
        }
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self { value: self.value / rhs, derivative: self.derivative / rhs }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    #[inline]
    fn div(self, rhs: Dual) -> Dual {
        let b2 = rhs.value * rhs.value;
        Dual {
            value: self / rhs.value,
            derivative: -(self * rhs.derivative) / b2,
        }
    }
}

// ============================================================================
// From conversions
// ============================================================================

impl From<f64> for Dual {
    #[inline]
    fn from(v: f64) -> Self {
        Dual::constant(v)
    }
}

impl From<f32> for Dual {
    #[inline]
    fn from(v: f32) -> Self {
        Dual::constant(f64::from(v))
    }
}

impl From<i32> for Dual {
    #[inline]
    fn from(v: i32) -> Self {
        Dual::constant(f64::from(v))
    }
}

impl From<i64> for Dual {
    #[inline]
    fn from(v: i64) -> Self {
        Dual::constant(v as f64)
    }
}

// ============================================================================
// PartialOrd — compare on primal values only
// ============================================================================

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// ============================================================================
// Sum / Product iterators
// ============================================================================

impl std::iter::Sum for Dual {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Dual::zero(), |acc, x| acc + x)
    }
}

impl std::iter::Product for Dual {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Dual::one(), |acc, x| acc * x)
    }
}

// ============================================================================
// HyperDual — second-order dual number (dual-of-dual)
// ============================================================================

/// A *hyper dual* number for second-order forward-mode AD.
///
/// Represents `f + f'·ε₁ + f'·ε₂ + f''·ε₁ε₂` where `ε₁² = ε₂² = 0` and
/// `ε₁ε₂ ≠ 0`.  This allows simultaneous computation of the function value,
/// two first-order derivatives (useful for mixed partials), and one second-order
/// mixed partial.
///
/// # Second-order diagonal (Hessian diagonal)
///
/// For the diagonal entry `∂²f/∂xᵢ²`, set `ε₁ = ε₂ = eᵢ` (the i-th basis
/// vector):
///
/// ```rust
/// use scirs2_autograd::functional::dual::HyperDual;
///
/// // f(x,y) = x^2 * y; ∂²f/∂x² = 2y at (3, 2) => 4
/// let x = HyperDual::variable(3.0);  // ε₁ = ε₂ = 1 for x
/// let y = HyperDual::constant(2.0);
/// let f = x * x * y;
/// assert!((f.value - 18.0).abs() < 1e-12);    // 9 * 2
/// assert!((f.d12 - 4.0).abs() < 1e-12);        // ∂²f/∂x² = 2y = 4
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperDual {
    /// Function value: `f`
    pub value: f64,
    /// First partial w.r.t. ε₁: `∂f/∂ε₁`
    pub d1: f64,
    /// First partial w.r.t. ε₂: `∂f/∂ε₂`
    pub d2: f64,
    /// Mixed second partial w.r.t. ε₁ε₂: `∂²f/∂ε₁∂ε₂`
    pub d12: f64,
}

impl fmt::Display for HyperDual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperDual({}, {}, {}, {})", self.value, self.d1, self.d2, self.d12)
    }
}

impl HyperDual {
    /// Construct with all components.
    #[inline]
    pub fn new(value: f64, d1: f64, d2: f64, d12: f64) -> Self {
        Self { value, d1, d2, d12 }
    }

    /// Constant — all derivative components are zero.
    #[inline]
    pub fn constant(value: f64) -> Self {
        Self { value, d1: 0.0, d2: 0.0, d12: 0.0 }
    }

    /// Active variable seeded for *both* ε₁ and ε₂ directions.
    ///
    /// Use this to compute the second-order diagonal `∂²f/∂xᵢ²` when
    /// evaluating `f` with `xᵢ` as the only active variable.
    #[inline]
    pub fn variable(value: f64) -> Self {
        Self { value, d1: 1.0, d2: 1.0, d12: 0.0 }
    }

    /// Active variable seeded for ε₁ only (d2 = 0).
    ///
    /// Use together with `variable2` to compute the off-diagonal entry
    /// `∂²f/∂xᵢ∂xⱼ`.
    #[inline]
    pub fn variable1(value: f64) -> Self {
        Self { value, d1: 1.0, d2: 0.0, d12: 0.0 }
    }

    /// Active variable seeded for ε₂ only (d1 = 0).
    #[inline]
    pub fn variable2(value: f64) -> Self {
        Self { value, d1: 0.0, d2: 1.0, d12: 0.0 }
    }

    /// Zero.
    #[inline]
    pub fn zero() -> Self {
        Self { value: 0.0, d1: 0.0, d2: 0.0, d12: 0.0 }
    }

    /// One (constant).
    #[inline]
    pub fn one() -> Self {
        Self::constant(1.0)
    }

    // -----------------------------------------------------------------------
    // Transcendentals
    // -----------------------------------------------------------------------

    /// `exp` of a hyper-dual number.
    pub fn exp(self) -> Self {
        let e = self.value.exp();
        Self {
            value: e,
            d1: e * self.d1,
            d2: e * self.d2,
            d12: e * (self.d12 + self.d1 * self.d2),
        }
    }

    /// `ln` of a hyper-dual number.
    pub fn ln(self) -> Self {
        let inv = 1.0 / self.value;
        let inv2 = -inv * inv;
        Self {
            value: self.value.ln(),
            d1: inv * self.d1,
            d2: inv * self.d2,
            d12: inv * self.d12 + inv2 * self.d1 * self.d2,
        }
    }

    /// `sqrt` of a hyper-dual number.
    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        let inv2s = 0.5 / s;
        let neg_inv4s3 = -0.25 / (s * s * s);
        Self {
            value: s,
            d1: inv2s * self.d1,
            d2: inv2s * self.d2,
            d12: inv2s * self.d12 + neg_inv4s3 * self.d1 * self.d2,
        }
    }

    /// `sin` of a hyper-dual number.
    pub fn sin(self) -> Self {
        let (sin_v, cos_v) = (self.value.sin(), self.value.cos());
        Self {
            value: sin_v,
            d1: cos_v * self.d1,
            d2: cos_v * self.d2,
            d12: cos_v * self.d12 - sin_v * self.d1 * self.d2,
        }
    }

    /// `cos` of a hyper-dual number.
    pub fn cos(self) -> Self {
        let (sin_v, cos_v) = (self.value.sin(), self.value.cos());
        Self {
            value: cos_v,
            d1: -sin_v * self.d1,
            d2: -sin_v * self.d2,
            d12: -sin_v * self.d12 - cos_v * self.d1 * self.d2,
        }
    }

    /// `tanh` of a hyper-dual number.
    pub fn tanh(self) -> Self {
        let t = self.value.tanh();
        let sech2 = 1.0 - t * t;
        let neg2_tanh_sech2 = -2.0 * t * sech2;
        Self {
            value: t,
            d1: sech2 * self.d1,
            d2: sech2 * self.d2,
            d12: sech2 * self.d12 + neg2_tanh_sech2 * self.d1 * self.d2,
        }
    }

    /// `powi` for hyper-dual numbers.
    pub fn powi(self, n: i32) -> Self {
        let nf = f64::from(n);
        let val_n = self.value.powi(n);
        let val_n1 = if n == 0 { 0.0 } else { self.value.powi(n - 1) };
        let val_n2 = if n <= 1 { 0.0 } else { self.value.powi(n - 2) };
        Self {
            value: val_n,
            d1: nf * val_n1 * self.d1,
            d2: nf * val_n1 * self.d2,
            d12: nf * val_n1 * self.d12 + nf * (nf - 1.0) * val_n2 * self.d1 * self.d2,
        }
    }

    /// `powf` for hyper-dual numbers.
    pub fn powf(self, p: f64) -> Self {
        let val_p = self.value.powf(p);
        let val_p1 = self.value.powf(p - 1.0);
        let val_p2 = self.value.powf(p - 2.0);
        Self {
            value: val_p,
            d1: p * val_p1 * self.d1,
            d2: p * val_p1 * self.d2,
            d12: p * val_p1 * self.d12 + p * (p - 1.0) * val_p2 * self.d1 * self.d2,
        }
    }
}

// Arithmetic for HyperDual
impl Neg for HyperDual {
    type Output = Self;
    fn neg(self) -> Self {
        Self { value: -self.value, d1: -self.d1, d2: -self.d2, d12: -self.d12 }
    }
}

impl Add for HyperDual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            d1: self.d1 + rhs.d1,
            d2: self.d2 + rhs.d2,
            d12: self.d12 + rhs.d12,
        }
    }
}

impl Add<f64> for HyperDual {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self { value: self.value + rhs, ..self }
    }
}

impl Sub for HyperDual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            d1: self.d1 - rhs.d1,
            d2: self.d2 - rhs.d2,
            d12: self.d12 - rhs.d12,
        }
    }
}

impl Mul for HyperDual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Product rule for hyper-duals:
        // (f + f₁ε₁ + f₂ε₂ + f₁₂ε₁ε₂)(g + g₁ε₁ + g₂ε₂ + g₁₂ε₁ε₂)
        // = fg + (f₁g + fg₁)ε₁ + (f₂g + fg₂)ε₂
        //      + (f₁₂g + f₁g₂ + f₂g₁ + fg₁₂)ε₁ε₂
        Self {
            value: self.value * rhs.value,
            d1: self.d1 * rhs.value + self.value * rhs.d1,
            d2: self.d2 * rhs.value + self.value * rhs.d2,
            d12: self.d12 * rhs.value
                + self.d1 * rhs.d2
                + self.d2 * rhs.d1
                + self.value * rhs.d12,
        }
    }
}

impl Mul<f64> for HyperDual {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            value: self.value * rhs,
            d1: self.d1 * rhs,
            d2: self.d2 * rhs,
            d12: self.d12 * rhs,
        }
    }
}

impl Div for HyperDual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // Quotient rule for hyper-duals
        let g = rhs.value;
        let g2 = g * g;
        let g3 = g2 * g;
        Self {
            value: self.value / g,
            d1: (self.d1 * g - self.value * rhs.d1) / g2,
            d2: (self.d2 * g - self.value * rhs.d2) / g2,
            d12: (self.d12 * g - self.value * rhs.d12) / g2
                - (self.d1 * rhs.d2 + self.d2 * rhs.d1) / g2
                + 2.0 * self.value * rhs.d1 * rhs.d2 / g3,
        }
    }
}

impl Div<f64> for HyperDual {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            value: self.value / rhs,
            d1: self.d1 / rhs,
            d2: self.d2 / rhs,
            d12: self.d12 / rhs,
        }
    }
}

impl From<f64> for HyperDual {
    fn from(v: f64) -> Self {
        HyperDual::constant(v)
    }
}

impl std::iter::Sum for HyperDual {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(HyperDual::zero(), |acc, x| acc + x)
    }
}

impl std::iter::Product for HyperDual {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(HyperDual::one(), |acc, x| acc * x)
    }
}

// ============================================================================
// Utility: evaluate a dual-number function and extract gradient
// ============================================================================

/// Evaluate `f` at `x` using dual numbers and return `(f(x), ∇f(x))`.
///
/// This performs **exactly n+1 forward passes** (one per partial derivative),
/// each as a single forward evaluation.  The function is generic over any `F`
/// that accepts `&[Dual]` and returns a single `Dual`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::dual::{Dual, eval_gradient};
///
/// let (val, grad) = eval_gradient(
///     |xs: &[Dual]| xs[0] * xs[0] + xs[1] * xs[1] + xs[0] * xs[1],
///     &[3.0, 4.0],
/// );
/// // f(3,4) = 9 + 16 + 12 = 37
/// assert!((val - 37.0).abs() < 1e-12);
/// // ∂f/∂x₀ = 2*3 + 4 = 10,  ∂f/∂x₁ = 2*4 + 3 = 11
/// assert!((grad[0] - 10.0).abs() < 1e-12);
/// assert!((grad[1] - 11.0).abs() < 1e-12);
/// ```
pub fn eval_gradient<F>(f: F, x: &[f64]) -> (f64, Vec<f64>)
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x.len();
    let mut grad = vec![0.0f64; n];
    let mut value = 0.0f64;

    for i in 0..n {
        let xs: Vec<Dual> = x
            .iter()
            .enumerate()
            .map(|(j, &xj)| {
                if j == i {
                    Dual::variable(xj)
                } else {
                    Dual::constant(xj)
                }
            })
            .collect();
        let out = f(&xs);
        if i == 0 {
            value = out.value;
        }
        grad[i] = out.derivative;
    }

    (value, grad)
}

/// Evaluate `f` at `x` using hyper-duals and return the full Hessian matrix.
///
/// Computes `H[i][j] = ∂²f/∂xᵢ∂xⱼ` for all `i, j` pairs.  Each off-diagonal
/// pair requires one hyper-dual forward evaluation; diagonal entries also
/// require one forward evaluation each.  Total cost: `n(n+1)/2` evaluations.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::dual::{HyperDual, eval_hessian};
///
/// // f(x,y) = x^2 + 3*x*y + 2*y^2
/// // H = [[2, 3], [3, 4]]
/// let h = eval_hessian(
///     |xs: &[HyperDual]| {
///         xs[0].powi(2) + HyperDual::constant(3.0) * xs[0] * xs[1]
///             + HyperDual::constant(2.0) * xs[1].powi(2)
///     },
///     &[1.0, 1.0],
/// );
/// assert!((h[0][0] - 2.0).abs() < 1e-10);
/// assert!((h[0][1] - 3.0).abs() < 1e-10);
/// assert!((h[1][0] - 3.0).abs() < 1e-10);
/// assert!((h[1][1] - 4.0).abs() < 1e-10);
/// ```
pub fn eval_hessian<F>(f: F, x: &[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[HyperDual]) -> HyperDual,
{
    let n = x.len();
    let mut h = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in i..n {
            let xs: Vec<HyperDual> = x
                .iter()
                .enumerate()
                .map(|(k, &xk)| {
                    if k == i && k == j {
                        HyperDual::variable(xk) // both ε₁ and ε₂ seeded
                    } else if k == i {
                        HyperDual::variable1(xk)
                    } else if k == j {
                        HyperDual::variable2(xk)
                    } else {
                        HyperDual::constant(xk)
                    }
                })
                .collect();
            let out = f(&xs);
            h[i][j] = out.d12;
            h[j][i] = out.d12; // Hessian is symmetric
        }
    }

    h
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_basic_ops() {
        let a = Dual::new(3.0, 1.0);
        let b = Dual::new(2.0, 0.0);

        // add
        let c = a + b;
        assert!((c.value - 5.0).abs() < 1e-12);
        assert!((c.derivative - 1.0).abs() < 1e-12);

        // mul
        let d = a * b;
        assert!((d.value - 6.0).abs() < 1e-12);
        assert!((d.derivative - 2.0).abs() < 1e-12); // product rule: 1*2 + 3*0 = 2

        // div
        let e = a / b;
        assert!((e.value - 1.5).abs() < 1e-12);
        assert!((e.derivative - 0.5).abs() < 1e-12); // (1*2 - 3*0)/4 = 0.5
    }

    #[test]
    fn test_dual_transcendentals() {
        let x = Dual::variable(0.0_f64);
        let y = x.exp();
        assert!((y.value - 1.0).abs() < 1e-12);
        assert!((y.derivative - 1.0).abs() < 1e-12);

        let z = Dual::variable(1.0_f64).ln();
        assert!((z.value - 0.0).abs() < 1e-12);
        assert!((z.derivative - 1.0).abs() < 1e-12);

        let s = Dual::variable(std::f64::consts::FRAC_PI_2).sin();
        assert!((s.value - 1.0).abs() < 1e-12);
        assert!(s.derivative.abs() < 1e-12); // cos(π/2) ≈ 0
    }

    #[test]
    fn test_eval_gradient_quadratic() {
        // f(x,y) = x^2 + y^2; ∇f = (2x, 2y)
        let (val, grad) = eval_gradient(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[3.0, 4.0]);
        assert!((val - 25.0).abs() < 1e-12);
        assert!((grad[0] - 6.0).abs() < 1e-12);
        assert!((grad[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_eval_gradient_mixed() {
        // f(x,y) = x*y; ∂f/∂x = y = 2, ∂f/∂y = x = 3
        let (val, grad) = eval_gradient(|xs| xs[0] * xs[1], &[3.0, 2.0]);
        assert!((val - 6.0).abs() < 1e-12);
        assert!((grad[0] - 2.0).abs() < 1e-12);
        assert!((grad[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_eval_hessian_quadratic() {
        // f(x,y) = x^2 + 3*x*y + 2*y^2; H = [[2,3],[3,4]]
        let h = eval_hessian(
            |xs| xs[0].powi(2) + HyperDual::constant(3.0) * xs[0] * xs[1]
                + HyperDual::constant(2.0) * xs[1].powi(2),
            &[1.0, 1.0],
        );
        assert!((h[0][0] - 2.0).abs() < 1e-10, "H[0][0]={}", h[0][0]);
        assert!((h[0][1] - 3.0).abs() < 1e-10, "H[0][1]={}", h[0][1]);
        assert!((h[1][0] - 3.0).abs() < 1e-10, "H[1][0]={}", h[1][0]);
        assert!((h[1][1] - 4.0).abs() < 1e-10, "H[1][1]={}", h[1][1]);
    }

    #[test]
    fn test_hyper_dual_exp() {
        // f(x) = exp(x); f''(x) = exp(x)
        let x = HyperDual::variable(1.0);
        let y = x.exp();
        let e = std::f64::consts::E;
        assert!((y.value - e).abs() < 1e-12);
        assert!((y.d1 - e).abs() < 1e-12);
        assert!((y.d2 - e).abs() < 1e-12);
        assert!((y.d12 - e).abs() < 1e-12);
    }

    #[test]
    fn test_dual_relu() {
        assert_eq!(Dual::new(2.0, 1.0).relu(), Dual::new(2.0, 1.0));
        assert_eq!(Dual::new(-1.0, 1.0).relu(), Dual::zero());
    }

    #[test]
    fn test_dual_sigmoid_derivative() {
        // d/dx sigmoid(x) = sigmoid(x)*(1 - sigmoid(x))
        let x = Dual::variable(0.0);
        let s = x.sigmoid();
        assert!((s.value - 0.5).abs() < 1e-12);
        assert!((s.derivative - 0.25).abs() < 1e-12); // 0.5*0.5 = 0.25
    }

    #[test]
    fn test_dual_display() {
        let d = Dual::new(3.0, -1.5);
        let s = format!("{}", d);
        assert!(s.contains("3") && s.contains("1.5"), "display: {}", s);
    }

    #[test]
    fn test_dual_from_primitives() {
        let d: Dual = 3.14_f64.into();
        assert!((d.value - 3.14).abs() < 1e-12);
        assert_eq!(d.derivative, 0.0);

        let di: Dual = 5_i32.into();
        assert!((di.value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_dual_clamp() {
        let d = Dual::new(0.5, 2.0).clamp(0.0, 1.0);
        assert!((d.value - 0.5).abs() < 1e-12);
        assert!((d.derivative - 2.0).abs() < 1e-12);

        let clamped = Dual::new(1.5, 2.0).clamp(0.0, 1.0);
        assert!((clamped.value - 1.0).abs() < 1e-12);
        assert_eq!(clamped.derivative, 0.0); // zeroed when clamped
    }
}
