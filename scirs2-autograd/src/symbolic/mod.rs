//! Symbolic differentiation capabilities
//!
//! This module provides symbolic representation and manipulation of mathematical expressions,
//! enabling symbolic differentiation and expression simplification.

use crate::{error::AutogradError, Float, Result};
use std::fmt;

pub mod expression;
pub mod rules;
pub mod simplify;
pub mod symbolic_tape;

/// Symbolic expression representation
#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr<T: Float> {
    /// Constant value
    Const(T),
    /// Variable with name
    Var(String),
    /// Addition: a + b
    Add(Box<SymExpr<T>>, Box<SymExpr<T>>),
    /// Subtraction: a - b
    Sub(Box<SymExpr<T>>, Box<SymExpr<T>>),
    /// Multiplication: a * b
    Mul(Box<SymExpr<T>>, Box<SymExpr<T>>),
    /// Division: a / b
    Div(Box<SymExpr<T>>, Box<SymExpr<T>>),
    /// Power: a ^ b
    Pow(Box<SymExpr<T>>, Box<SymExpr<T>>),
    /// Exponential: exp(a)
    Exp(Box<SymExpr<T>>),
    /// Natural logarithm: ln(a)
    Log(Box<SymExpr<T>>),
    /// Sine: sin(a)
    Sin(Box<SymExpr<T>>),
    /// Cosine: cos(a)
    Cos(Box<SymExpr<T>>),
    /// Hyperbolic tangent: tanh(a)
    Tanh(Box<SymExpr<T>>),
}

impl<T: Float> SymExpr<T> {
    /// Create a constant expression
    pub fn constant(value: T) -> Self {
        Self::Const(value)
    }

    /// Create a variable expression
    pub fn variable(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Differentiate with respect to a variable
    pub fn differentiate(&self, var: &str) -> Self {
        match self {
            SymExpr::Const(_) => SymExpr::Const(T::zero()),
            SymExpr::Var(name) => {
                if name == var {
                    SymExpr::Const(T::one())
                } else {
                    SymExpr::Const(T::zero())
                }
            }
            SymExpr::Add(a, b) => SymExpr::Add(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var)),
            ),
            SymExpr::Sub(a, b) => SymExpr::Sub(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var)),
            ),
            SymExpr::Mul(a, b) => {
                // Product rule: (a*b)' = a'*b + a*b'
                SymExpr::Add(
                    Box::new(SymExpr::Mul(Box::new(a.differentiate(var)), b.clone())),
                    Box::new(SymExpr::Mul(a.clone(), Box::new(b.differentiate(var)))),
                )
            }
            SymExpr::Div(a, b) => {
                // Quotient rule: (a/b)' = (a'*b - a*b') / b²
                SymExpr::Div(
                    Box::new(SymExpr::Sub(
                        Box::new(SymExpr::Mul(Box::new(a.differentiate(var)), b.clone())),
                        Box::new(SymExpr::Mul(a.clone(), Box::new(b.differentiate(var)))),
                    )),
                    Box::new(SymExpr::Pow(
                        b.clone(),
                        Box::new(SymExpr::Const(T::from(2).expect("Convert 2"))),
                    )),
                )
            }
            SymExpr::Pow(a, b) => {
                // Power rule with chain rule: (a^b)' = b*a^(b-1)*a' (for constant b)
                // General case: (a^b)' = a^b * (b'*ln(a) + b*a'/a)
                SymExpr::Mul(
                    self.clone().into(),
                    Box::new(SymExpr::Add(
                        Box::new(SymExpr::Mul(
                            Box::new(b.differentiate(var)),
                            Box::new(SymExpr::Log(a.clone())),
                        )),
                        Box::new(SymExpr::Mul(
                            b.clone(),
                            Box::new(SymExpr::Div(Box::new(a.differentiate(var)), a.clone())),
                        )),
                    )),
                )
            }
            SymExpr::Exp(a) => {
                // (exp(a))' = exp(a) * a'
                SymExpr::Mul(self.clone().into(), Box::new(a.differentiate(var)))
            }
            SymExpr::Log(a) => {
                // (ln(a))' = a' / a
                SymExpr::Div(Box::new(a.differentiate(var)), a.clone())
            }
            SymExpr::Sin(a) => {
                // (sin(a))' = cos(a) * a'
                SymExpr::Mul(
                    Box::new(SymExpr::Cos(a.clone())),
                    Box::new(a.differentiate(var)),
                )
            }
            SymExpr::Cos(a) => {
                // (cos(a))' = -sin(a) * a'
                SymExpr::Mul(
                    Box::new(SymExpr::Mul(
                        Box::new(SymExpr::Const(T::from(-1).expect("Convert -1"))),
                        Box::new(SymExpr::Sin(a.clone())),
                    )),
                    Box::new(a.differentiate(var)),
                )
            }
            SymExpr::Tanh(a) => {
                // (tanh(a))' = (1 - tanh²(a)) * a'
                SymExpr::Mul(
                    Box::new(SymExpr::Sub(
                        Box::new(SymExpr::Const(T::one())),
                        Box::new(SymExpr::Pow(
                            self.clone().into(),
                            Box::new(SymExpr::Const(T::from(2).expect("Convert 2"))),
                        )),
                    )),
                    Box::new(a.differentiate(var)),
                )
            }
        }
    }

    /// Simplify the expression
    pub fn simplify(&self) -> Self {
        simplify::simplify_expr(self)
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, T>) -> Result<T> {
        match self {
            SymExpr::Const(c) => Ok(*c),
            SymExpr::Var(name) => vars.get(name).copied().ok_or_else(|| {
                AutogradError::invalid_argument(format!("Variable '{}' not found", name))
            }),
            SymExpr::Add(a, b) => Ok(a.evaluate(vars)? + b.evaluate(vars)?),
            SymExpr::Sub(a, b) => Ok(a.evaluate(vars)? - b.evaluate(vars)?),
            SymExpr::Mul(a, b) => Ok(a.evaluate(vars)? * b.evaluate(vars)?),
            SymExpr::Div(a, b) => {
                let b_val = b.evaluate(vars)?;
                if b_val.abs() < T::epsilon() {
                    return Err(AutogradError::compute_error("Division by zero".to_string()));
                }
                Ok(a.evaluate(vars)? / b_val)
            }
            SymExpr::Pow(a, b) => Ok(a.evaluate(vars)?.powf(b.evaluate(vars)?)),
            SymExpr::Exp(a) => Ok(a.evaluate(vars)?.exp()),
            SymExpr::Log(a) => {
                let a_val = a.evaluate(vars)?;
                if a_val <= T::zero() {
                    return Err(AutogradError::compute_error(
                        "Logarithm of non-positive number".to_string(),
                    ));
                }
                Ok(a_val.ln())
            }
            SymExpr::Sin(a) => Ok(a.evaluate(vars)?.sin()),
            SymExpr::Cos(a) => Ok(a.evaluate(vars)?.cos()),
            SymExpr::Tanh(a) => Ok(a.evaluate(vars)?.tanh()),
        }
    }
}

impl<T: Float> fmt::Display for SymExpr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymExpr::Const(c) => write!(f, "{}", c),
            SymExpr::Var(name) => write!(f, "{}", name),
            SymExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            SymExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            SymExpr::Mul(a, b) => write!(f, "({} * {})", a, b),
            SymExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            SymExpr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            SymExpr::Exp(a) => write!(f, "exp({})", a),
            SymExpr::Log(a) => write!(f, "ln({})", a),
            SymExpr::Sin(a) => write!(f, "sin({})", a),
            SymExpr::Cos(a) => write!(f, "cos({})", a),
            SymExpr::Tanh(a) => write!(f, "tanh({})", a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_symbolic_constant() {
        let expr: SymExpr<f64> = SymExpr::constant(5.0);
        let deriv = expr.differentiate("x");

        let mut vars = HashMap::new();
        assert_eq!(deriv.evaluate(&vars).expect("Should evaluate"), 0.0);
    }

    #[test]
    fn test_symbolic_variable() {
        let expr: SymExpr<f64> = SymExpr::variable("x");
        let deriv = expr.differentiate("x");

        let mut vars = HashMap::new();
        assert_eq!(deriv.evaluate(&vars).expect("Should evaluate"), 1.0);
    }

    #[test]
    fn test_symbolic_addition() {
        // f(x) = x + 2
        let x: SymExpr<f64> = SymExpr::variable("x");
        let two = SymExpr::constant(2.0);
        let expr = SymExpr::Add(Box::new(x), Box::new(two));

        // f'(x) = 1
        let deriv = expr.differentiate("x");

        let mut vars = HashMap::new();
        assert_eq!(deriv.evaluate(&vars).expect("Should evaluate"), 1.0);
    }

    #[test]
    fn test_symbolic_multiplication() {
        // f(x) = 3 * x
        let x: SymExpr<f64> = SymExpr::variable("x");
        let three = SymExpr::constant(3.0);
        let expr = SymExpr::Mul(Box::new(three), Box::new(x));

        // f'(x) = 3
        let deriv = expr.differentiate("x").simplify();

        let mut vars = HashMap::new();
        assert_eq!(deriv.evaluate(&vars).expect("Should evaluate"), 3.0);
    }

    #[test]
    fn test_symbolic_power() {
        // f(x) = x²
        let x: SymExpr<f64> = SymExpr::variable("x");
        let two = SymExpr::constant(2.0);
        let expr = SymExpr::Pow(Box::new(x.clone()), Box::new(two));

        // f'(x) = 2x
        let deriv = expr.differentiate("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);

        // At x=3, derivative should be 6
        let result = deriv.evaluate(&vars).expect("Should evaluate");
        assert!((result - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_display() {
        let x: SymExpr<f64> = SymExpr::variable("x");
        let two = SymExpr::constant(2.0);
        let expr = SymExpr::Add(Box::new(x), Box::new(two));

        assert_eq!(format!("{}", expr), "(x + 2)");
    }
}
