//! Symbolic Probability Expressions for Do-Calculus
//!
//! Provides a recursive algebra for representing and pretty-printing
//! interventional distributions such as P(Y | do(X), Z).
//!
//! # Expression types
//!
//! | Variant | Notation |
//! |---------|---------|
//! | `Joint` | P(X₁, X₂, ...) |
//! | `Conditional` | P(num) / P(den) |
//! | `Marginal` | Σ_{vars} expr |
//! | `Interventional` | P(Y | do(X)) |
//! | `Product` | ∏ exprs |
//! | `Quotient` | num / den |
//!
//! # References
//!
//! - Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
//!   Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.

use std::fmt;

// ---------------------------------------------------------------------------
// ProbExpr
// ---------------------------------------------------------------------------

/// A symbolic probability expression in the do-calculus.
///
/// Can represent interventional and observational distributions,
/// products, quotients, and marginalizations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ProbExpr {
    /// P(X₁, X₂, ...) — joint probability over named variables.
    Joint(Vec<String>),

    /// P(numerator | conditioning) — conditional probability.
    /// Stored as numerator expression conditioned on denominator expression.
    Conditional {
        /// The expression in the numerator: P(Y, Z | ...)
        numerator: Box<ProbExpr>,
        /// The denominator expression (the conditioning part).
        denominator: Box<ProbExpr>,
    },

    /// Σ_{summand_vars} expr — marginalization over `summand_vars`.
    Marginal {
        /// The expression to marginalize.
        expr: Box<ProbExpr>,
        /// Variables being summed out.
        summand_vars: Vec<String>,
    },

    /// P(Y | do(x)) — interventional distribution.
    /// The `do_vars` are the intervened variables.
    Interventional {
        /// The expression under intervention (the outcome distribution).
        expr: Box<ProbExpr>,
        /// Variables being set by intervention: do(X₁, X₂, ...).
        do_vars: Vec<String>,
    },

    /// ∏ exprs — product of multiple probability expressions.
    Product(Vec<ProbExpr>),

    /// num / den — ratio of two probability expressions.
    Quotient {
        /// Numerator expression.
        num: Box<ProbExpr>,
        /// Denominator expression.
        den: Box<ProbExpr>,
    },
}

// ---------------------------------------------------------------------------
// Display implementation
// ---------------------------------------------------------------------------

impl fmt::Display for ProbExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbExpr::Joint(vars) => {
                if vars.is_empty() {
                    write!(f, "P()")
                } else {
                    write!(f, "P({})", vars.join(", "))
                }
            }

            ProbExpr::Conditional {
                numerator,
                denominator,
            } => {
                // Render as P(num_vars | den_vars), extracting variable names when possible
                match (numerator.as_ref(), denominator.as_ref()) {
                    (ProbExpr::Joint(num_vars), ProbExpr::Joint(den_vars)) => {
                        write!(f, "P({} | {})", num_vars.join(", "), den_vars.join(", "))
                    }
                    _ => {
                        write!(f, "[{numerator}] / [{denominator}]")
                    }
                }
            }

            ProbExpr::Marginal { expr, summand_vars } => {
                if summand_vars.is_empty() {
                    write!(f, "{expr}")
                } else {
                    write!(f, "Σ_{{{vars}}} {expr}", vars = summand_vars.join(", "))
                }
            }

            ProbExpr::Interventional { expr, do_vars } => {
                if do_vars.is_empty() {
                    write!(f, "{expr}")
                } else {
                    // Flatten the inner expr if it's a Joint to produce P(Y | do(X))
                    match expr.as_ref() {
                        ProbExpr::Joint(outcome_vars) => {
                            write!(
                                f,
                                "P({} | do({}))",
                                outcome_vars.join(", "),
                                do_vars.join(", ")
                            )
                        }
                        _ => {
                            write!(f, "P({expr} | do({do}))", do = do_vars.join(", "))
                        }
                    }
                }
            }

            ProbExpr::Product(exprs) => {
                if exprs.is_empty() {
                    write!(f, "1")
                } else if exprs.len() == 1 {
                    write!(f, "{}", exprs[0])
                } else {
                    let parts: Vec<String> = exprs.iter().map(|e| format!("{e}")).collect();
                    write!(f, "{}", parts.join(" · "))
                }
            }

            ProbExpr::Quotient { num, den } => {
                write!(f, "[{num}] / [{den}]")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

impl ProbExpr {
    /// Simplify an expression:
    /// - Remove trivial marginals (summand_vars is empty)
    /// - Flatten nested products
    /// - Remove unit products (empty Product → 1, single-element Product)
    /// - Recursively simplify sub-expressions
    pub fn simplify(&self) -> ProbExpr {
        match self {
            ProbExpr::Marginal { expr, summand_vars } => {
                let inner = expr.simplify();
                if summand_vars.is_empty() {
                    inner
                } else {
                    ProbExpr::Marginal {
                        expr: Box::new(inner),
                        summand_vars: summand_vars.clone(),
                    }
                }
            }

            ProbExpr::Product(exprs) => {
                // Flatten nested products and simplify recursively
                let mut flat: Vec<ProbExpr> = Vec::new();
                for e in exprs {
                    let s = e.simplify();
                    match s {
                        ProbExpr::Product(sub) => flat.extend(sub),
                        other => flat.push(other),
                    }
                }
                // Remove unit factors (would need numeric evaluation — skip for symbolic)
                if flat.is_empty() {
                    // Empty product = 1 represented as P() (certain event)
                    ProbExpr::Joint(Vec::new())
                } else if flat.len() == 1 {
                    flat.remove(0)
                } else {
                    ProbExpr::Product(flat)
                }
            }

            ProbExpr::Conditional {
                numerator,
                denominator,
            } => ProbExpr::Conditional {
                numerator: Box::new(numerator.simplify()),
                denominator: Box::new(denominator.simplify()),
            },

            ProbExpr::Interventional { expr, do_vars } => {
                let inner = expr.simplify();
                if do_vars.is_empty() {
                    inner
                } else {
                    ProbExpr::Interventional {
                        expr: Box::new(inner),
                        do_vars: do_vars.clone(),
                    }
                }
            }

            ProbExpr::Quotient { num, den } => ProbExpr::Quotient {
                num: Box::new(num.simplify()),
                den: Box::new(den.simplify()),
            },

            // Leaf nodes: no simplification needed
            ProbExpr::Joint(_) => self.clone(),
        }
    }

    /// Construct a simple interventional expression P(y_vars | do(x_vars)).
    pub fn p_do(y_vars: Vec<String>, x_vars: Vec<String>) -> Self {
        ProbExpr::Interventional {
            expr: Box::new(ProbExpr::Joint(y_vars)),
            do_vars: x_vars,
        }
    }

    /// Construct P(vars) — simple joint probability.
    pub fn p(vars: Vec<String>) -> Self {
        ProbExpr::Joint(vars)
    }

    /// Construct a marginal Σ_{summand_vars} expr.
    pub fn marginal(expr: ProbExpr, summand_vars: Vec<String>) -> Self {
        ProbExpr::Marginal {
            expr: Box::new(expr),
            summand_vars,
        }
    }

    /// Construct a conditional probability P(y | z) where both y and z are Joint.
    pub fn conditional(y_vars: Vec<String>, z_vars: Vec<String>) -> Self {
        ProbExpr::Conditional {
            numerator: Box::new(ProbExpr::Joint(y_vars)),
            denominator: Box::new(ProbExpr::Joint(z_vars)),
        }
    }

    /// Construct a product of expressions.
    pub fn product(exprs: Vec<ProbExpr>) -> Self {
        ProbExpr::Product(exprs)
    }

    /// Collect the top-level variable names if this is a Joint expression.
    pub fn joint_vars(&self) -> Option<&[String]> {
        match self {
            ProbExpr::Joint(vars) => Some(vars),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn s(s: &str) -> String {
        s.to_owned()
    }

    #[test]
    fn test_display_joint() {
        let e = ProbExpr::p(vec![s("Y")]);
        assert_eq!(format!("{e}"), "P(Y)");
    }

    #[test]
    fn test_display_joint_multi() {
        let e = ProbExpr::p(vec![s("X"), s("Y"), s("Z")]);
        assert_eq!(format!("{e}"), "P(X, Y, Z)");
    }

    #[test]
    fn test_display_conditional() {
        let e = ProbExpr::conditional(vec![s("Y")], vec![s("X")]);
        assert_eq!(format!("{e}"), "P(Y | X)");
    }

    #[test]
    fn test_display_interventional() {
        let e = ProbExpr::p_do(vec![s("Y")], vec![s("X")]);
        assert_eq!(format!("{e}"), "P(Y | do(X))");
    }

    #[test]
    fn test_display_interventional_multiple_do() {
        let e = ProbExpr::p_do(vec![s("Y")], vec![s("X1"), s("X2")]);
        assert_eq!(format!("{e}"), "P(Y | do(X1, X2))");
    }

    #[test]
    fn test_display_marginal() {
        let inner = ProbExpr::p(vec![s("Y"), s("Z")]);
        let e = ProbExpr::marginal(inner, vec![s("Z")]);
        assert_eq!(format!("{e}"), "Σ_{Z} P(Y, Z)");
    }

    #[test]
    fn test_display_product() {
        let e1 = ProbExpr::p(vec![s("X")]);
        let e2 = ProbExpr::p(vec![s("Y")]);
        let prod = ProbExpr::product(vec![e1, e2]);
        assert_eq!(format!("{prod}"), "P(X) · P(Y)");
    }

    #[test]
    fn test_display_empty_product() {
        let prod = ProbExpr::product(vec![]);
        // Simplification of empty product
        let simplified = prod.simplify();
        // Should collapse to P() (certain event)
        assert!(matches!(simplified, ProbExpr::Joint(ref v) if v.is_empty()));
    }

    #[test]
    fn test_simplify_trivial_marginal() {
        let inner = ProbExpr::p(vec![s("Y")]);
        let marginal = ProbExpr::marginal(inner.clone(), vec![]);
        let simplified = marginal.simplify();
        assert_eq!(simplified, inner);
    }

    #[test]
    fn test_simplify_nested_product_flattening() {
        let e1 = ProbExpr::p(vec![s("X")]);
        let e2 = ProbExpr::p(vec![s("Y")]);
        let e3 = ProbExpr::p(vec![s("Z")]);
        let inner_prod = ProbExpr::product(vec![e1.clone(), e2.clone()]);
        let outer_prod = ProbExpr::product(vec![inner_prod, e3.clone()]);
        let simplified = outer_prod.simplify();
        match simplified {
            ProbExpr::Product(ref terms) => {
                assert_eq!(terms.len(), 3);
            }
            _ => panic!("Expected Product with 3 terms, got: {simplified:?}"),
        }
    }

    #[test]
    fn test_simplify_single_element_product() {
        let e = ProbExpr::p(vec![s("Y")]);
        let prod = ProbExpr::product(vec![e.clone()]);
        let simplified = prod.simplify();
        assert_eq!(simplified, e);
    }

    #[test]
    fn test_conditional_display_complex() {
        // P(Y | X, Z)
        let e = ProbExpr::conditional(vec![s("Y")], vec![s("X"), s("Z")]);
        assert_eq!(format!("{e}"), "P(Y | X, Z)");
    }

    #[test]
    fn test_interventional_with_marginal() {
        // Σ_M P(Y | do(X), M)
        let inner = ProbExpr::Interventional {
            expr: Box::new(ProbExpr::Joint(vec![s("Y")])),
            do_vars: vec![s("X")],
        };
        let marg = ProbExpr::marginal(inner, vec![s("M")]);
        let disp = format!("{marg}");
        assert!(disp.contains("Σ_{M}"), "Should contain summation: {disp}");
        assert!(disp.contains("do(X)"), "Should contain do(X): {disp}");
    }
}
