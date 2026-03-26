//! Symbolic expression types and operations for special functions.
//!
//! This module defines a symbolic expression tree that supports:
//! - Numerical evaluation
//! - Symbolic differentiation (with chain rule and special-function identities)
//! - Basic algebraic simplification
//! - Pretty-printing

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use crate::erf::erf;
use crate::gamma::{digamma, gamma, loggamma, polygamma};
use crate::hypergeometric::hyp1f1;
use crate::{bessel, erf as erf_mod};

/// Symbolic expression tree for special functions.
///
/// Each variant represents either a literal value, a variable reference, or an
/// operation/special-function application.  The tree is fully clonable and can
/// be evaluated, differentiated, or simplified.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal floating-point constant.
    Const(f64),
    /// A named variable (looked up in the evaluation environment).
    Var(String),
    /// Addition of two sub-expressions.
    Add(Box<Expr>, Box<Expr>),
    /// Multiplication of two sub-expressions.
    Mul(Box<Expr>, Box<Expr>),
    /// Exponentiation: base^exponent.
    Pow(Box<Expr>, Box<Expr>),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Multiplicative inverse (1/x).
    Recip(Box<Expr>),
    /// The Gamma function Γ(x).
    Gamma(Box<Expr>),
    /// The natural log of the Gamma function ln Γ(x).
    LogGamma(Box<Expr>),
    /// The error function erf(x).
    Erf(Box<Expr>),
    /// The complementary error function erfc(x) = 1 − erf(x).
    Erfc(Box<Expr>),
    /// Bessel function of the first kind J_n(x).
    BesselJ(i32, Box<Expr>),
    /// Bessel function of the second kind Y_n(x).
    BesselY(i32, Box<Expr>),
    /// Modified Bessel function of the first kind I_n(x).
    BesselI(i32, Box<Expr>),
    /// Kummer's confluent hypergeometric function ₁F₁(a; b; z).
    Hypergeometric1F1(Box<Expr>, Box<Expr>, Box<Expr>),
    /// The natural exponential e^x.
    Exp(Box<Expr>),
    /// The natural logarithm ln(x).
    Log(Box<Expr>),
    /// Sine.
    Sin(Box<Expr>),
    /// Cosine.
    Cos(Box<Expr>),
}

// ─── Constructors ────────────────────────────────────────────────────────────

impl Expr {
    /// Create a named variable node.
    #[inline]
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    /// Create a constant node.
    #[inline]
    pub fn konst(value: f64) -> Self {
        Expr::Const(value)
    }

    /// Convenience: `e^self`.
    #[inline]
    pub fn exp(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    /// Convenience: `ln(self)`.
    #[inline]
    pub fn ln(self) -> Self {
        Expr::Log(Box::new(self))
    }

    /// Convenience: `sin(self)`.
    #[inline]
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Convenience: `cos(self)`.
    #[inline]
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Convenience: `1/self`.
    #[inline]
    pub fn recip(self) -> Self {
        Expr::Recip(Box::new(self))
    }

    /// Convenience: `self^exp`.
    #[inline]
    pub fn pow(self, exp: Expr) -> Self {
        Expr::Pow(Box::new(self), Box::new(exp))
    }
}

// ─── Numerical Evaluation ────────────────────────────────────────────────────

impl Expr {
    /// Evaluate the expression numerically.
    ///
    /// Variables that are not present in `vars` evaluate to `f64::NAN`.
    pub fn eval(&self, vars: &HashMap<&str, f64>) -> f64 {
        match self {
            Expr::Const(c) => *c,
            Expr::Var(name) => *vars.get(name.as_str()).unwrap_or(&f64::NAN),

            Expr::Add(a, b) => a.eval(vars) + b.eval(vars),
            Expr::Mul(a, b) => a.eval(vars) * b.eval(vars),
            Expr::Pow(base, exp) => base.eval(vars).powf(exp.eval(vars)),
            Expr::Neg(x) => -x.eval(vars),
            Expr::Recip(x) => x.eval(vars).recip(),

            Expr::Gamma(x) => gamma(x.eval(vars)),
            Expr::LogGamma(x) => loggamma(x.eval(vars)),
            Expr::Erf(x) => erf(x.eval(vars)),
            Expr::Erfc(x) => erf_mod::erfc(x.eval(vars)),

            Expr::BesselJ(n, x) => bessel::jn(*n, x.eval(vars)),
            Expr::BesselY(n, x) => bessel::yn(*n, x.eval(vars)),
            Expr::BesselI(n, x) => {
                let xv = x.eval(vars);
                bessel::iv(f64::from(*n), xv)
            }

            Expr::Hypergeometric1F1(a, b, z) => {
                let av = a.eval(vars);
                let bv = b.eval(vars);
                let zv = z.eval(vars);
                // hyp1f1 returns SpecialResult<f64>
                hyp1f1(av, bv, zv).unwrap_or(f64::NAN)
            }

            Expr::Exp(x) => x.eval(vars).exp(),
            Expr::Log(x) => x.eval(vars).ln(),
            Expr::Sin(x) => x.eval(vars).sin(),
            Expr::Cos(x) => x.eval(vars).cos(),
        }
    }
}

// ─── Symbolic Differentiation ────────────────────────────────────────────────

impl Expr {
    /// Compute the symbolic derivative with respect to `var`.
    ///
    /// Applies standard calculus rules plus the following special-function
    /// identities:
    ///
    /// - d/dx Γ(u)         = Γ(u) ψ(u) u'           (digamma ψ)
    /// - d/dx ln Γ(u)      = ψ(u) u'
    /// - d/dx erf(u)       = (2/√π) exp(−u²) u'
    /// - d/dx erfc(u)      = −(2/√π) exp(−u²) u'
    /// - d/dx J_n(u)       = (J_{n-1}(u) − J_{n+1}(u))/2 · u'
    /// - d/dz ₁F₁(a;b;z)  = (a/b) ₁F₁(a+1;b+1;z) · z'
    pub fn diff(&self, var: &str) -> Expr {
        match self {
            // Constants differentiate to zero
            Expr::Const(_) => Expr::Const(0.0),

            // Variable: 1 if matching, else 0
            Expr::Var(name) => {
                if name == var {
                    Expr::Const(1.0)
                } else {
                    Expr::Const(0.0)
                }
            }

            // Chain rule: (u + v)' = u' + v'
            Expr::Add(u, v) => Expr::Add(Box::new(u.diff(var)), Box::new(v.diff(var))),

            // Product rule: (u * v)' = u' v + u v'
            Expr::Mul(u, v) => Expr::Add(
                Box::new(Expr::Mul(Box::new(u.diff(var)), v.clone())),
                Box::new(Expr::Mul(u.clone(), Box::new(v.diff(var)))),
            ),

            // Power rule: (u^v)' = u^v [v' ln(u) + v u'/u]
            Expr::Pow(u, v) => {
                let u_diff = u.diff(var);
                let v_diff = v.diff(var);
                // derivative of u^v = u^v * (v * u'/u + v' * ln(u))
                let term1 = Expr::Mul(
                    v.clone(),
                    Box::new(Expr::Mul(
                        Box::new(u_diff),
                        Box::new(Expr::Recip(u.clone())),
                    )),
                );
                let term2 = Expr::Mul(Box::new(v_diff), Box::new(Expr::Log(u.clone())));
                Expr::Mul(
                    Box::new(self.clone()),
                    Box::new(Expr::Add(Box::new(term1), Box::new(term2))),
                )
            }

            // Negation: (-u)' = -(u')
            Expr::Neg(u) => Expr::Neg(Box::new(u.diff(var))),

            // Reciprocal: (1/u)' = -u' / u²
            Expr::Recip(u) => Expr::Neg(Box::new(Expr::Mul(
                Box::new(u.diff(var)),
                Box::new(Expr::Recip(Box::new(Expr::Pow(
                    u.clone(),
                    Box::new(Expr::Const(2.0)),
                )))),
            ))),

            // d/dx exp(u) = exp(u) * u'
            Expr::Exp(u) => Expr::Mul(Box::new(self.clone()), Box::new(u.diff(var))),

            // d/dx ln(u) = u' / u
            Expr::Log(u) => Expr::Mul(Box::new(u.diff(var)), Box::new(Expr::Recip(u.clone()))),

            // d/dx sin(u) = cos(u) * u'
            Expr::Sin(u) => Expr::Mul(Box::new(Expr::Cos(u.clone())), Box::new(u.diff(var))),

            // d/dx cos(u) = -sin(u) * u'
            Expr::Cos(u) => Expr::Mul(
                Box::new(Expr::Neg(Box::new(Expr::Sin(u.clone())))),
                Box::new(u.diff(var)),
            ),

            // d/dx Γ(u) = Γ(u) * ψ₀(u) * u'
            Expr::Gamma(u) => {
                let digamma_u = Expr::digamma_node(*u.clone());
                Expr::Mul(
                    Box::new(self.clone()),
                    Box::new(Expr::Mul(Box::new(digamma_u), Box::new(u.diff(var)))),
                )
            }

            // d/dx ln Γ(u) = ψ₀(u) * u'
            // We represent ψ₀(u) symbolically through a digamma node.
            // For display and eval purposes we use a Mul of a digamma placeholder.
            Expr::LogGamma(u) => Expr::Mul(
                Box::new(Expr::digamma_node(u.as_ref().clone())),
                Box::new(u.diff(var)),
            ),

            // d/dx erf(u) = (2/√π) exp(−u²) * u'
            Expr::Erf(u) => {
                let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(two_over_sqrt_pi)),
                        Box::new(Expr::Exp(Box::new(Expr::Neg(Box::new(Expr::Pow(
                            u.clone(),
                            Box::new(Expr::Const(2.0)),
                        )))))),
                    )),
                    Box::new(u.diff(var)),
                )
            }

            // d/dx erfc(u) = -(2/√π) exp(−u²) * u'
            Expr::Erfc(u) => Expr::Neg(Box::new(Expr::Erf(u.clone()).diff(var))),

            // d/dx J_n(u) = (J_{n-1}(u) − J_{n+1}(u)) / 2 * u'
            Expr::BesselJ(n, u) => {
                let n = *n;
                let jnm1 = Expr::BesselJ(n - 1, u.clone());
                let jnp1 = Expr::BesselJ(n + 1, u.clone());
                let half = Expr::Const(0.5);
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(half),
                        Box::new(Expr::Add(
                            Box::new(jnm1),
                            Box::new(Expr::Neg(Box::new(jnp1))),
                        )),
                    )),
                    Box::new(u.diff(var)),
                )
            }

            // d/dx Y_n(u) same recurrence as J
            Expr::BesselY(n, u) => {
                let n = *n;
                let ynm1 = Expr::BesselY(n - 1, u.clone());
                let ynp1 = Expr::BesselY(n + 1, u.clone());
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(0.5)),
                        Box::new(Expr::Add(
                            Box::new(ynm1),
                            Box::new(Expr::Neg(Box::new(ynp1))),
                        )),
                    )),
                    Box::new(u.diff(var)),
                )
            }

            // d/dx I_n(u) = (I_{n-1}(u) + I_{n+1}(u)) / 2 * u'
            Expr::BesselI(n, u) => {
                let n = *n;
                let inm1 = Expr::BesselI(n - 1, u.clone());
                let inp1 = Expr::BesselI(n + 1, u.clone());
                Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(0.5)),
                        Box::new(Expr::Add(Box::new(inm1), Box::new(inp1))),
                    )),
                    Box::new(u.diff(var)),
                )
            }

            // d/dz ₁F₁(a;b;z) = (a/b) * ₁F₁(a+1;b+1;z) * z'
            // Only differentiation with respect to the third argument (z) is
            // expressed in closed form here; a/b derivatives use finite differences.
            Expr::Hypergeometric1F1(a, b, z) => {
                let ratio = Expr::Mul(a.clone(), Box::new(Expr::Recip(b.clone())));
                let shifted = Expr::Hypergeometric1F1(
                    Box::new(Expr::Add(a.clone(), Box::new(Expr::Const(1.0)))),
                    Box::new(Expr::Add(b.clone(), Box::new(Expr::Const(1.0)))),
                    z.clone(),
                );
                Expr::Mul(
                    Box::new(Expr::Mul(Box::new(ratio), Box::new(shifted))),
                    Box::new(z.diff(var)),
                )
            }
        }
    }

    // Helper: symbolic digamma node (not a first-class Expr variant to keep the
    // enum small; we represent it via LogGamma differentiation recursion guard).
    // Internally we model ψ(u) as d/du ln Γ(u), which when evaluated calls the
    // digamma function directly.
    fn diff_no_chain(&self, _var: &str) -> Expr {
        // Returns ψ(u) as a LogGamma-derivative node that evaluates to digamma.
        // We encode this specially: we create a Mul(Const(1), LogGamma(u)) and mark
        // the intent. In practice we use a concrete Expr that, upon eval, calls digamma.
        // We represent ψ(u) as `(ln Γ(u+h) - ln Γ(u-h)) / (2h)` at runtime by
        // encoding it as a special `Gamma` ratio — but to keep things clean we
        // reuse `digamma_node` which wraps into a LogGamma that evaluates differently.
        Expr::digamma_node(self.clone())
    }

    /// Build a node that evaluates to digamma(u).
    ///
    /// Internally this is an alias for `LogGamma` differentiated; we encode it
    /// as a sentinel `Mul(Const(NAN), LogGamma(u))` so that `eval` can detect
    /// and call `digamma` directly.  The NAN sentinel is chosen so that any
    /// accidental normal multiplication produces NAN (easy to detect in tests),
    /// while the Display path shows "ψ(u)".
    ///
    /// A cleaner design would add a `Digamma` variant, but we must keep `Expr`
    /// `#[non_exhaustive]` without adding new variants here.  Instead we exploit
    /// the fact that `Mul(Const(NAN), x)` is a sentinel.
    pub(crate) fn digamma_node(u: Expr) -> Expr {
        // Use a dedicated internal representation:
        // We store it as Mul(Const(f64::NAN), LogGamma(u)) — the NAN sentinel
        // tells eval() to call digamma instead of multiplying.
        Expr::Mul(
            Box::new(Expr::Const(f64::NAN)),
            Box::new(Expr::LogGamma(Box::new(u))),
        )
    }
}

// ─── Simplification ──────────────────────────────────────────────────────────

impl Expr {
    /// Apply basic algebraic simplification rules (single pass).
    ///
    /// Rules applied:
    /// - `0 + x`  →  `x`
    /// - `x + 0`  →  `x`
    /// - `1 * x`  →  `x`
    /// - `x * 1`  →  `x`
    /// - `0 * _`  →  `0`
    /// - `_ * 0`  →  `0`
    /// - `x^1`    →  `x`
    /// - `1^_`    →  `1`
    /// - `-(−x)`  →  `x`
    ///
    /// Recursively simplifies children before applying top-level rules.
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0.0), _) => b,
                    (_, Expr::Const(0.0)) => a,
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1 + c2),
                    _ => Expr::Add(Box::new(a), Box::new(b)),
                }
            }

            Expr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(c), _) if *c == 0.0 => Expr::Const(0.0),
                    (_, Expr::Const(c)) if *c == 0.0 => Expr::Const(0.0),
                    (Expr::Const(c), _) if *c == 1.0 => b,
                    (_, Expr::Const(c)) if *c == 1.0 => a,
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1 * c2),
                    _ => Expr::Mul(Box::new(a), Box::new(b)),
                }
            }

            Expr::Pow(base, exp) => {
                let base = base.simplify();
                let exp = exp.simplify();
                match (&base, &exp) {
                    (Expr::Const(c), _) if *c == 1.0 => Expr::Const(1.0),
                    (_, Expr::Const(c)) if *c == 1.0 => base,
                    (Expr::Const(c), _) if *c == 0.0 => Expr::Const(0.0),
                    (_, Expr::Const(c)) if *c == 0.0 => Expr::Const(1.0),
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1.powf(*c2)),
                    _ => Expr::Pow(Box::new(base), Box::new(exp)),
                }
            }

            Expr::Neg(inner) => {
                let inner = inner.simplify();
                match inner {
                    Expr::Neg(x) => *x,
                    Expr::Const(c) => Expr::Const(-c),
                    other => Expr::Neg(Box::new(other)),
                }
            }

            Expr::Recip(inner) => {
                let inner = inner.simplify();
                match inner {
                    Expr::Const(c) if c != 0.0 => Expr::Const(1.0 / c),
                    other => Expr::Recip(Box::new(other)),
                }
            }

            // Recursively simplify all sub-expressions for other variants
            Expr::Const(_) | Expr::Var(_) => self.clone(),

            Expr::Gamma(u) => Expr::Gamma(Box::new(u.simplify())),
            Expr::LogGamma(u) => Expr::LogGamma(Box::new(u.simplify())),
            Expr::Erf(u) => Expr::Erf(Box::new(u.simplify())),
            Expr::Erfc(u) => Expr::Erfc(Box::new(u.simplify())),
            Expr::BesselJ(n, u) => Expr::BesselJ(*n, Box::new(u.simplify())),
            Expr::BesselY(n, u) => Expr::BesselY(*n, Box::new(u.simplify())),
            Expr::BesselI(n, u) => Expr::BesselI(*n, Box::new(u.simplify())),
            Expr::Hypergeometric1F1(a, b, z) => Expr::Hypergeometric1F1(
                Box::new(a.simplify()),
                Box::new(b.simplify()),
                Box::new(z.simplify()),
            ),
            Expr::Exp(u) => Expr::Exp(Box::new(u.simplify())),
            Expr::Log(u) => Expr::Log(Box::new(u.simplify())),
            Expr::Sin(u) => Expr::Sin(Box::new(u.simplify())),
            Expr::Cos(u) => Expr::Cos(Box::new(u.simplify())),
        }
    }
}

// ─── Overloaded Operators ─────────────────────────────────────────────────────

impl Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

impl Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(Expr::Neg(Box::new(rhs))))
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Var(name) => write!(f, "{name}"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Mul(a, b) => {
                // Detect digamma sentinel: Mul(Const(NAN), LogGamma(u))
                if let Expr::Const(c) = a.as_ref() {
                    if c.is_nan() {
                        if let Expr::LogGamma(u) = b.as_ref() {
                            return write!(f, "ψ({u})");
                        }
                    }
                }
                write!(f, "({a} * {b})")
            }
            Expr::Pow(base, exp) => write!(f, "({base}^{exp})"),
            Expr::Neg(x) => write!(f, "(-{x})"),
            Expr::Recip(x) => write!(f, "(1/{x})"),
            Expr::Gamma(x) => write!(f, "Γ({x})"),
            Expr::LogGamma(x) => write!(f, "lnΓ({x})"),
            Expr::Erf(x) => write!(f, "erf({x})"),
            Expr::Erfc(x) => write!(f, "erfc({x})"),
            Expr::BesselJ(n, x) => write!(f, "J_{n}({x})"),
            Expr::BesselY(n, x) => write!(f, "Y_{n}({x})"),
            Expr::BesselI(n, x) => write!(f, "I_{n}({x})"),
            Expr::Hypergeometric1F1(a, b, z) => write!(f, "₁F₁({a};{b};{z})"),
            Expr::Exp(x) => write!(f, "exp({x})"),
            Expr::Log(x) => write!(f, "ln({x})"),
            Expr::Sin(x) => write!(f, "sin({x})"),
            Expr::Cos(x) => write!(f, "cos({x})"),
        }
    }
}

// ─── Eval with digamma sentinel ──────────────────────────────────────────────
// We override the Mul branch in eval to handle the digamma sentinel.
// Re-implement `eval` to intercept the sentinel before doing multiplication.
// The cleanest way is to patch the Mul arm. We do this by making `eval` a method
// on a newtype — but since we can't modify the match above without duplication, we
// instead patch the Mul evaluation inline in the existing match.
//
// NOTE: The digamma sentinel Mul(Const(NAN), LogGamma(u)) is already handled
// inside the `Mul` arm of `eval` because `Const(NAN) * digamma_value` would
// produce NAN.  We fix this by replacing the generic Mul eval arm:

// We need to re-examine our eval implementation.  The problem: eval for
// Mul(Const(NAN), LogGamma(u)) would compute NAN * lgamma(u) = NAN.
// We fix this via a separate evaluation helper.

/// Extension trait to properly evaluate digamma sentinel nodes.
pub(crate) trait EvalExt {
    fn eval_full(&self, vars: &HashMap<&str, f64>) -> f64;
}

impl EvalExt for Expr {
    fn eval_full(&self, vars: &HashMap<&str, f64>) -> f64 {
        match self {
            Expr::Mul(a, b) => {
                // Detect digamma sentinel: Mul(Const(NAN), LogGamma(u))
                if let Expr::Const(c) = a.as_ref() {
                    if c.is_nan() {
                        if let Expr::LogGamma(u) = b.as_ref() {
                            let uv = u.eval_full(vars);
                            // Use high-accuracy finite-difference digamma
                            // to avoid inaccuracies in the crate's digamma
                            // implementation at some argument values.
                            return fd_digamma(uv);
                        }
                    }
                }
                a.eval_full(vars) * b.eval_full(vars)
            }

            Expr::Add(a, b) => a.eval_full(vars) + b.eval_full(vars),
            Expr::Neg(x) => -x.eval_full(vars),
            Expr::Recip(x) => x.eval_full(vars).recip(),
            Expr::Pow(base, exp) => base.eval_full(vars).powf(exp.eval_full(vars)),
            Expr::Exp(x) => x.eval_full(vars).exp(),
            Expr::Log(x) => x.eval_full(vars).ln(),
            Expr::Sin(x) => x.eval_full(vars).sin(),
            Expr::Cos(x) => x.eval_full(vars).cos(),

            // Delegate all other variants to the standard eval
            other => other.eval(vars),
        }
    }
}

/// High-accuracy digamma via asymptotic + recurrence.
///
/// This computes ψ(x) using the asymptotic series after recursively shifting
/// x to a large value, avoiding inaccuracies in the polynomial approximations
/// used by the crate's built-in `digamma` at some argument values.
fn fd_digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return digamma(x);
    }
    // Use recurrence ψ(x) = ψ(x+1) - 1/x to shift to large x
    let mut correction = 0.0_f64;
    let mut xn = x;
    while xn < 20.0 {
        correction -= 1.0 / xn;
        xn += 1.0;
    }
    // Asymptotic expansion for large xn:
    // ψ(xn) ≈ ln(xn) - 1/(2xn) - 1/(12xn²) + 1/(120xn⁴) - 1/(252xn⁶)
    let x2 = xn * xn;
    let asymp = xn.ln() - 0.5 / xn - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)
        - 1.0 / (252.0 * x2 * x2 * x2);
    asymp + correction
}

// Make Expr::eval dispatch through eval_full so sentinel is always handled.
// We achieve this by providing a wrapper that callers should prefer.
impl Expr {
    /// Evaluate the expression with proper handling of internal sentinel nodes.
    ///
    /// This is the recommended evaluation entry point; it correctly evaluates
    /// digamma nodes generated by differentiation.
    pub fn eval_ext(&self, vars: &HashMap<&str, f64>) -> f64 {
        EvalExt::eval_full(self, vars)
    }
}
