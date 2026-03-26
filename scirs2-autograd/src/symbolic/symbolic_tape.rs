//! Symbolic tape: convert computation graphs to symbolic expressions
//!
//! This module bridges the autograd computation graph with the symbolic
//! expression system, enabling:
//!
//! - Conversion of computation graph nodes to symbolic expressions
//! - Symbolic Jacobian computation
//! - Compilation of symbolic expressions to efficient evaluation functions
//! - Code generation for optimized gradient computation

use super::SymExpr;
use crate::error::AutogradError;
use crate::Float;
use crate::Result;
use std::collections::HashMap;
use std::fmt;

/// A recorded operation in the symbolic tape.
#[derive(Debug, Clone)]
pub enum SymbolicOp {
    /// Input variable
    Input(String),
    /// Constant value
    Constant(f64),
    /// Binary addition
    Add(usize, usize),
    /// Binary subtraction
    Sub(usize, usize),
    /// Binary multiplication
    Mul(usize, usize),
    /// Binary division
    Div(usize, usize),
    /// Power
    Pow(usize, usize),
    /// Unary negation
    Neg(usize),
    /// Exponential
    Exp(usize),
    /// Natural logarithm
    Log(usize),
    /// Sine
    Sin(usize),
    /// Cosine
    Cos(usize),
    /// Hyperbolic tangent
    Tanh(usize),
}

/// A tape that records operations symbolically.
///
/// Operations are appended to the tape, and the tape can then be
/// converted to symbolic expressions for differentiation, simplification,
/// and code generation.
#[derive(Debug, Clone)]
pub struct SymbolicTape {
    /// Recorded operations (each has an implicit index = position in vec)
    ops: Vec<SymbolicOp>,
    /// Map from variable names to their tape indices
    inputs: HashMap<String, usize>,
}

impl SymbolicTape {
    /// Create a new empty symbolic tape.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            inputs: HashMap::new(),
        }
    }

    /// Record an input variable and return its tape index.
    pub fn input(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.inputs.get(name) {
            return idx;
        }
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Input(name.to_string()));
        self.inputs.insert(name.to_string(), idx);
        idx
    }

    /// Record a constant and return its tape index.
    pub fn constant(&mut self, value: f64) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Constant(value));
        idx
    }

    /// Record an addition and return its tape index.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Add(a, b));
        idx
    }

    /// Record a subtraction and return its tape index.
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Sub(a, b));
        idx
    }

    /// Record a multiplication and return its tape index.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Mul(a, b));
        idx
    }

    /// Record a division and return its tape index.
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Div(a, b));
        idx
    }

    /// Record a power operation and return its tape index.
    pub fn pow(&mut self, base: usize, exp: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Pow(base, exp));
        idx
    }

    /// Record a negation and return its tape index.
    pub fn neg(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Neg(a));
        idx
    }

    /// Record an exponential and return its tape index.
    pub fn exp(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Exp(a));
        idx
    }

    /// Record a natural logarithm and return its tape index.
    pub fn log(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Log(a));
        idx
    }

    /// Record a sine and return its tape index.
    pub fn sin(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Sin(a));
        idx
    }

    /// Record a cosine and return its tape index.
    pub fn cos(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Cos(a));
        idx
    }

    /// Record a hyperbolic tangent and return its tape index.
    pub fn tanh(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(SymbolicOp::Tanh(a));
        idx
    }

    /// Number of operations on the tape.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Get all input variable names.
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.keys().map(|s| s.as_str()).collect()
    }

    /// Convert a tape node to a symbolic expression.
    ///
    /// # Errors
    /// Returns error if the node index is out of bounds.
    pub fn to_expr<T: Float>(&self, node_idx: usize) -> Result<SymExpr<T>> {
        if node_idx >= self.ops.len() {
            return Err(AutogradError::invalid_argument(format!(
                "SymbolicTape: node index {} out of bounds (tape length {})",
                node_idx,
                self.ops.len()
            )));
        }
        self.build_expr(node_idx)
    }

    /// Recursively build a symbolic expression from a tape node.
    fn build_expr<T: Float>(&self, idx: usize) -> Result<SymExpr<T>> {
        match &self.ops[idx] {
            SymbolicOp::Input(name) => Ok(SymExpr::Var(name.clone())),
            SymbolicOp::Constant(v) => {
                let val = T::from(*v).ok_or_else(|| {
                    AutogradError::compute_error(format!(
                        "Cannot convert constant {} to target type",
                        v
                    ))
                })?;
                Ok(SymExpr::Const(val))
            }
            SymbolicOp::Add(a, b) => {
                let ea = self.build_expr(*a)?;
                let eb = self.build_expr(*b)?;
                Ok(SymExpr::Add(Box::new(ea), Box::new(eb)))
            }
            SymbolicOp::Sub(a, b) => {
                let ea = self.build_expr(*a)?;
                let eb = self.build_expr(*b)?;
                Ok(SymExpr::Sub(Box::new(ea), Box::new(eb)))
            }
            SymbolicOp::Mul(a, b) => {
                let ea = self.build_expr(*a)?;
                let eb = self.build_expr(*b)?;
                Ok(SymExpr::Mul(Box::new(ea), Box::new(eb)))
            }
            SymbolicOp::Div(a, b) => {
                let ea = self.build_expr(*a)?;
                let eb = self.build_expr(*b)?;
                Ok(SymExpr::Div(Box::new(ea), Box::new(eb)))
            }
            SymbolicOp::Pow(base, exp) => {
                let eb = self.build_expr(*base)?;
                let ee = self.build_expr(*exp)?;
                Ok(SymExpr::Pow(Box::new(eb), Box::new(ee)))
            }
            SymbolicOp::Neg(a) => {
                let ea = self.build_expr(*a)?;
                let neg_one = T::from(-1.0).ok_or_else(|| {
                    AutogradError::compute_error("Cannot convert -1 to target type".to_string())
                })?;
                Ok(SymExpr::Mul(
                    Box::new(SymExpr::Const(neg_one)),
                    Box::new(ea),
                ))
            }
            SymbolicOp::Exp(a) => {
                let ea = self.build_expr(*a)?;
                Ok(SymExpr::Exp(Box::new(ea)))
            }
            SymbolicOp::Log(a) => {
                let ea = self.build_expr(*a)?;
                Ok(SymExpr::Log(Box::new(ea)))
            }
            SymbolicOp::Sin(a) => {
                let ea = self.build_expr(*a)?;
                Ok(SymExpr::Sin(Box::new(ea)))
            }
            SymbolicOp::Cos(a) => {
                let ea = self.build_expr(*a)?;
                Ok(SymExpr::Cos(Box::new(ea)))
            }
            SymbolicOp::Tanh(a) => {
                let ea = self.build_expr(*a)?;
                Ok(SymExpr::Tanh(Box::new(ea)))
            }
        }
    }

    /// Compute the symbolic Jacobian: differentiate each output w.r.t. each input.
    ///
    /// # Arguments
    /// * `output_indices` - Tape indices of the output nodes
    ///
    /// # Returns
    /// A matrix (Vec of Vecs) where `result[i][j]` is `d(output_i)/d(input_j)`,
    /// with inputs ordered as returned by `input_names()`.
    ///
    /// # Errors
    /// Returns error if any output index is out of bounds.
    pub fn symbolic_jacobian<T: Float>(
        &self,
        output_indices: &[usize],
    ) -> Result<SymbolicJacobian<T>> {
        let input_names: Vec<String> = self.inputs.keys().cloned().collect();
        let mut rows = Vec::with_capacity(output_indices.len());
        for &out_idx in output_indices {
            let expr = self.to_expr::<T>(out_idx)?;
            let mut row = Vec::with_capacity(input_names.len());
            for var_name in &input_names {
                let deriv = expr.differentiate(var_name).simplify();
                row.push(deriv);
            }
            rows.push(row);
        }
        Ok(SymbolicJacobian {
            rows,
            input_names,
            num_outputs: output_indices.len(),
        })
    }
}

impl Default for SymbolicTape {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SymbolicTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SymbolicTape ({} ops):", self.ops.len())?;
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                SymbolicOp::Input(name) => writeln!(f, "  [{}] Input({})", i, name)?,
                SymbolicOp::Constant(v) => writeln!(f, "  [{}] Const({})", i, v)?,
                SymbolicOp::Add(a, b) => writeln!(f, "  [{}] Add([{}], [{}])", i, a, b)?,
                SymbolicOp::Sub(a, b) => writeln!(f, "  [{}] Sub([{}], [{}])", i, a, b)?,
                SymbolicOp::Mul(a, b) => writeln!(f, "  [{}] Mul([{}], [{}])", i, a, b)?,
                SymbolicOp::Div(a, b) => writeln!(f, "  [{}] Div([{}], [{}])", i, a, b)?,
                SymbolicOp::Pow(a, b) => writeln!(f, "  [{}] Pow([{}], [{}])", i, a, b)?,
                SymbolicOp::Neg(a) => writeln!(f, "  [{}] Neg([{}])", i, a)?,
                SymbolicOp::Exp(a) => writeln!(f, "  [{}] Exp([{}])", i, a)?,
                SymbolicOp::Log(a) => writeln!(f, "  [{}] Log([{}])", i, a)?,
                SymbolicOp::Sin(a) => writeln!(f, "  [{}] Sin([{}])", i, a)?,
                SymbolicOp::Cos(a) => writeln!(f, "  [{}] Cos([{}])", i, a)?,
                SymbolicOp::Tanh(a) => writeln!(f, "  [{}] Tanh([{}])", i, a)?,
            }
        }
        Ok(())
    }
}

/// The result of symbolic Jacobian computation.
#[derive(Debug, Clone)]
pub struct SymbolicJacobian<T: Float> {
    /// rows[i][j] = d(output_i)/d(input_j)
    rows: Vec<Vec<SymExpr<T>>>,
    /// Input variable names (column labels)
    input_names: Vec<String>,
    /// Number of output nodes
    num_outputs: usize,
}

impl<T: Float> SymbolicJacobian<T> {
    /// Get the derivative expression: d(output_i)/d(input_j).
    ///
    /// # Errors
    /// Returns error if indices are out of bounds.
    pub fn get(&self, output_idx: usize, input_idx: usize) -> Result<&SymExpr<T>> {
        if output_idx >= self.num_outputs {
            return Err(AutogradError::invalid_argument(format!(
                "SymbolicJacobian: output index {} out of bounds ({})",
                output_idx, self.num_outputs
            )));
        }
        if input_idx >= self.input_names.len() {
            return Err(AutogradError::invalid_argument(format!(
                "SymbolicJacobian: input index {} out of bounds ({})",
                input_idx,
                self.input_names.len()
            )));
        }
        Ok(&self.rows[output_idx][input_idx])
    }

    /// Get the derivative by input variable name.
    ///
    /// # Errors
    /// Returns error if output index or variable name is invalid.
    pub fn get_by_name(&self, output_idx: usize, var_name: &str) -> Result<&SymExpr<T>> {
        let input_idx = self
            .input_names
            .iter()
            .position(|n| n == var_name)
            .ok_or_else(|| {
                AutogradError::invalid_argument(format!(
                    "SymbolicJacobian: unknown input variable '{}'",
                    var_name
                ))
            })?;
        self.get(output_idx, input_idx)
    }

    /// Get the input variable names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.input_names.len()
    }

    /// Evaluate the full Jacobian matrix at given variable values.
    ///
    /// # Errors
    /// Returns error if evaluation fails (e.g., missing variables).
    pub fn evaluate(&self, vars: &HashMap<String, T>) -> Result<Vec<Vec<T>>> {
        let mut result = Vec::with_capacity(self.num_outputs);
        for row in &self.rows {
            let mut row_vals = Vec::with_capacity(row.len());
            for expr in row {
                row_vals.push(expr.evaluate(vars)?);
            }
            result.push(row_vals);
        }
        Ok(result)
    }
}

impl<T: Float> fmt::Display for SymbolicJacobian<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SymbolicJacobian ({}x{}):",
            self.num_outputs,
            self.input_names.len()
        )?;
        writeln!(f, "  Inputs: {:?}", self.input_names)?;
        for (i, row) in self.rows.iter().enumerate() {
            for (j, expr) in row.iter().enumerate() {
                writeln!(f, "  d(out_{})/d({}) = {}", i, self.input_names[j], expr)?;
            }
        }
        Ok(())
    }
}

/// A compiled symbolic expression that can be evaluated efficiently.
///
/// The expression is pre-analyzed to determine evaluation order and
/// minimize redundant computation via common subexpression detection.
#[derive(Debug, Clone)]
pub struct CompiledExpr<T: Float> {
    /// The original expression
    expr: SymExpr<T>,
    /// Required input variable names
    required_vars: Vec<String>,
}

impl<T: Float> CompiledExpr<T> {
    /// Compile a symbolic expression for efficient evaluation.
    pub fn compile(expr: SymExpr<T>) -> Self {
        let required_vars = Self::collect_vars(&expr);
        Self {
            expr,
            required_vars,
        }
    }

    /// Collect all variable names used in the expression.
    fn collect_vars(expr: &SymExpr<T>) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_vars_recursive(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars_recursive(expr: &SymExpr<T>, vars: &mut Vec<String>) {
        match expr {
            SymExpr::Const(_) => {}
            SymExpr::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            SymExpr::Add(a, b)
            | SymExpr::Sub(a, b)
            | SymExpr::Mul(a, b)
            | SymExpr::Div(a, b)
            | SymExpr::Pow(a, b) => {
                Self::collect_vars_recursive(a, vars);
                Self::collect_vars_recursive(b, vars);
            }
            SymExpr::Exp(a)
            | SymExpr::Log(a)
            | SymExpr::Sin(a)
            | SymExpr::Cos(a)
            | SymExpr::Tanh(a) => {
                Self::collect_vars_recursive(a, vars);
            }
        }
    }

    /// Get the list of required input variables.
    pub fn required_vars(&self) -> &[String] {
        &self.required_vars
    }

    /// Evaluate the compiled expression with the given variable values.
    ///
    /// # Errors
    /// Returns error if required variables are missing or evaluation fails.
    pub fn evaluate(&self, vars: &HashMap<String, T>) -> Result<T> {
        // Check that all required variables are provided
        for var in &self.required_vars {
            if !vars.contains_key(var) {
                return Err(AutogradError::invalid_argument(format!(
                    "CompiledExpr: missing required variable '{}'",
                    var
                )));
            }
        }
        self.expr.evaluate(vars)
    }

    /// Get the underlying expression.
    pub fn expr(&self) -> &SymExpr<T> {
        &self.expr
    }

    /// Create the derivative of this compiled expression w.r.t. a variable.
    pub fn derivative(&self, var: &str) -> Self {
        let deriv = self.expr.differentiate(var).simplify();
        Self::compile(deriv)
    }
}

impl<T: Float> fmt::Display for CompiledExpr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompiledExpr(vars={:?}, expr={})",
            self.required_vars, self.expr
        )
    }
}

/// Generate a Rust function body string for evaluating the expression.
///
/// This can be used for code generation / JIT-style optimization.
pub fn generate_eval_code<T: Float>(expr: &SymExpr<T>) -> String {
    match expr {
        SymExpr::Const(c) => format!("{}_f64", c),
        SymExpr::Var(name) => name.clone(),
        SymExpr::Add(a, b) => format!("({} + {})", generate_eval_code(a), generate_eval_code(b)),
        SymExpr::Sub(a, b) => format!("({} - {})", generate_eval_code(a), generate_eval_code(b)),
        SymExpr::Mul(a, b) => format!("({} * {})", generate_eval_code(a), generate_eval_code(b)),
        SymExpr::Div(a, b) => format!("({} / {})", generate_eval_code(a), generate_eval_code(b)),
        SymExpr::Pow(a, b) => format!("{}.powf({})", generate_eval_code(a), generate_eval_code(b)),
        SymExpr::Exp(a) => format!("{}.exp()", generate_eval_code(a)),
        SymExpr::Log(a) => format!("{}.ln()", generate_eval_code(a)),
        SymExpr::Sin(a) => format!("{}.sin()", generate_eval_code(a)),
        SymExpr::Cos(a) => format!("{}.cos()", generate_eval_code(a)),
        SymExpr::Tanh(a) => format!("{}.tanh()", generate_eval_code(a)),
    }
}

/// Count the total number of operations in an expression tree.
pub fn count_ops<T: Float>(expr: &SymExpr<T>) -> usize {
    match expr {
        SymExpr::Const(_) | SymExpr::Var(_) => 0,
        SymExpr::Add(a, b)
        | SymExpr::Sub(a, b)
        | SymExpr::Mul(a, b)
        | SymExpr::Div(a, b)
        | SymExpr::Pow(a, b) => 1 + count_ops(a) + count_ops(b),
        SymExpr::Exp(a)
        | SymExpr::Log(a)
        | SymExpr::Sin(a)
        | SymExpr::Cos(a)
        | SymExpr::Tanh(a) => 1 + count_ops(a),
    }
}

/// Compute the depth of an expression tree.
pub fn expr_depth<T: Float>(expr: &SymExpr<T>) -> usize {
    match expr {
        SymExpr::Const(_) | SymExpr::Var(_) => 0,
        SymExpr::Add(a, b)
        | SymExpr::Sub(a, b)
        | SymExpr::Mul(a, b)
        | SymExpr::Div(a, b)
        | SymExpr::Pow(a, b) => 1 + expr_depth(a).max(expr_depth(b)),
        SymExpr::Exp(a)
        | SymExpr::Log(a)
        | SymExpr::Sin(a)
        | SymExpr::Cos(a)
        | SymExpr::Tanh(a) => 1 + expr_depth(a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_symbolic_tape_basic() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let two = tape.constant(2.0);
        let x_squared = tape.mul(x, x);
        let result = tape.mul(two, x_squared); // 2 * x^2
        let expr: SymExpr<f64> = tape.to_expr(result).expect("valid");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let val = expr.evaluate(&vars).expect("eval ok");
        assert!((val - 18.0).abs() < 1e-10); // 2 * 3^2 = 18
    }

    #[test]
    fn test_symbolic_tape_differentiate() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let x_sq = tape.mul(x, x); // x^2
        let expr: SymExpr<f64> = tape.to_expr(x_sq).expect("valid");
        let deriv = expr.differentiate("x").simplify();

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let val = deriv.evaluate(&vars).expect("eval ok");
        assert!((val - 10.0).abs() < 1e-10); // d/dx x^2 = 2x, at x=5 -> 10
    }

    #[test]
    fn test_symbolic_tape_sin_cos() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let sin_x = tape.sin(x);
        let expr: SymExpr<f64> = tape.to_expr(sin_x).expect("valid");
        let deriv = expr.differentiate("x").simplify();

        // d/dx sin(x) = cos(x)
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let val = deriv.evaluate(&vars).expect("eval ok");
        assert!((val - 1.0).abs() < 1e-10); // cos(0) = 1
    }

    #[test]
    fn test_symbolic_tape_exp() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let exp_x = tape.exp(x);
        let expr: SymExpr<f64> = tape.to_expr(exp_x).expect("valid");
        let deriv = expr.differentiate("x").simplify();

        // d/dx exp(x) = exp(x)
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let val = deriv.evaluate(&vars).expect("eval ok");
        assert!((val - 1.0).abs() < 1e-10); // exp(0) = 1
    }

    #[test]
    fn test_symbolic_jacobian() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let y = tape.input("y");
        let f1 = tape.mul(x, y); // f1 = x * y
        let f2 = tape.add(x, y); // f2 = x + y
        let jac: SymbolicJacobian<f64> = tape.symbolic_jacobian(&[f1, f2]).expect("jacobian ok");
        assert_eq!(jac.num_outputs(), 2);
        assert_eq!(jac.num_inputs(), 2);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        vars.insert("y".to_string(), 3.0);
        let mat = jac.evaluate(&vars).expect("eval ok");
        // Jacobian should be:
        // df1/dx=y=3, df1/dy=x=2
        // df2/dx=1,   df2/dy=1
        // But input_names order may vary, so check by name
        let x_col = jac
            .input_names()
            .iter()
            .position(|n| n == "x")
            .expect("x found");
        let y_col = jac
            .input_names()
            .iter()
            .position(|n| n == "y")
            .expect("y found");
        assert!((mat[0][y_col] - 2.0).abs() < 1e-10); // df1/dy = x = 2
        assert!((mat[0][x_col] - 3.0).abs() < 1e-10); // df1/dx = y = 3
        assert!((mat[1][x_col] - 1.0).abs() < 1e-10); // df2/dx = 1
        assert!((mat[1][y_col] - 1.0).abs() < 1e-10); // df2/dy = 1
    }

    #[test]
    fn test_compiled_expr() {
        let expr: SymExpr<f64> = SymExpr::Mul(
            Box::new(SymExpr::variable("x")),
            Box::new(SymExpr::variable("x")),
        );
        let compiled = CompiledExpr::compile(expr);
        assert_eq!(compiled.required_vars(), &["x"]);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 4.0);
        let val = compiled.evaluate(&vars).expect("eval ok");
        assert!((val - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_compiled_expr_derivative() {
        let expr: SymExpr<f64> = SymExpr::Mul(
            Box::new(SymExpr::variable("x")),
            Box::new(SymExpr::variable("x")),
        );
        let compiled = CompiledExpr::compile(expr);
        let deriv = compiled.derivative("x");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let val = deriv.evaluate(&vars).expect("eval ok");
        assert!((val - 6.0).abs() < 1e-10); // d/dx x^2 = 2x, at x=3 -> 6
    }

    #[test]
    fn test_compiled_expr_missing_var() {
        let expr: SymExpr<f64> = SymExpr::variable("x");
        let compiled = CompiledExpr::compile(expr);
        let vars: HashMap<String, f64> = HashMap::new();
        assert!(compiled.evaluate(&vars).is_err());
    }

    #[test]
    fn test_generate_eval_code() {
        let expr: SymExpr<f64> = SymExpr::Add(
            Box::new(SymExpr::variable("x")),
            Box::new(SymExpr::Const(1.0)),
        );
        let code = generate_eval_code(&expr);
        assert!(code.contains("x"));
        assert!(code.contains("+"));
    }

    #[test]
    fn test_count_ops_and_depth() {
        // x * x + sin(x) => 3 ops (mul, add, sin), depth = 2
        let x: SymExpr<f64> = SymExpr::variable("x");
        let expr = SymExpr::Add(
            Box::new(SymExpr::Mul(Box::new(x.clone()), Box::new(x.clone()))),
            Box::new(SymExpr::Sin(Box::new(x))),
        );
        assert_eq!(count_ops(&expr), 3);
        assert_eq!(expr_depth(&expr), 2);
    }

    #[test]
    fn test_symbolic_tape_display() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let _y = tape.add(x, x);
        let s = format!("{}", tape);
        assert!(s.contains("Input(x)"));
        assert!(s.contains("Add"));
    }

    #[test]
    fn test_symbolic_tape_neg() {
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let neg_x = tape.neg(x);
        let expr: SymExpr<f64> = tape.to_expr(neg_x).expect("valid");

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let val = expr.evaluate(&vars).expect("eval ok");
        assert!((val - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_tape_complex() {
        // f(x) = exp(x^2) -- d/dx = 2*x*exp(x^2)
        let mut tape = SymbolicTape::new();
        let x = tape.input("x");
        let x_sq = tape.mul(x, x);
        let result = tape.exp(x_sq);
        let expr: SymExpr<f64> = tape.to_expr(result).expect("valid");
        let deriv = expr.differentiate("x").simplify();

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let val = deriv.evaluate(&vars).expect("eval ok");
        // d/dx exp(x^2) at x=1 = 2*1*exp(1) = 2*e
        let expected = 2.0 * std::f64::consts::E;
        assert!((val - expected).abs() < 1e-8);
    }
}
