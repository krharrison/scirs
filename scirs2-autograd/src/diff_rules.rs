//! Automatic differentiation rules — differentiable ops with VJP (reverse-mode)
//!
//! This module provides building blocks for graph-based automatic differentiation.
//! Each operation implements the [`DiffOp`] trait, which exposes both a forward
//! computation and the corresponding vector-Jacobian product (VJP) used in
//! reverse-mode AD (backpropagation).
//!
//! # Architecture
//!
//! ```text
//! DiffOp trait
//!   ├── forward(&self, inputs) -> Vec<f64>          (primal)
//!   └── vjp(&self, inputs, output_grad) -> Vec<Vec<f64>>  (backward)
//! ```
//!
//! # Implemented operations
//!
//! | Op | Forward | VJP |
//! |----|---------|-----|
//! | [`Add`] | z = x + y | dL/dx = dL/dz, dL/dy = dL/dz |
//! | [`Sub`] | z = x − y | dL/dx = dL/dz, dL/dy = −dL/dz |
//! | [`Mul`] | z = x ⊙ y | dL/dx = dL/dz ⊙ y, dL/dy = dL/dz ⊙ x |
//! | [`Div`] | z = x / y | standard quotient rule |
//! | [`Neg`] | z = −x | dL/dx = −dL/dz |
//! | [`Exp`] | z = exp(x) | dL/dx = dL/dz ⊙ exp(x) |
//! | [`Log`] | z = log(x) | dL/dx = dL/dz / x |
//! | [`Sqrt`] | z = √x | dL/dx = dL/dz / (2√x) |
//! | [`Relu`] | z = max(0, x) | dL/dx = dL/dz ⊙ 𝟙(x > 0) |
//! | [`Sigmoid`] | z = σ(x) | dL/dx = dL/dz ⊙ z(1−z) |
//! | [`Tanh`] | z = tanh(x) | dL/dx = dL/dz ⊙ (1 − tanh²(x)) |
//! | [`Softmax`] | z_i = exp(x_i)/Σexp | full Jacobian VJP |
//! | [`Sum`] | z = Σx | dL/dx = broadcast(dL/dz) |
//! | [`Scale`] | z = α·x | dL/dx = α·dL/dz |
//! | [`MatMul`] | Z = A @ B | dA = dZ @ Bᵀ, dB = Aᵀ @ dZ |
//! | [`Pow`] | z = xⁿ | dL/dx = n·xⁿ⁻¹·dL/dz |
//!
//! # Examples
//!
//! ```rust
//! use scirs2_autograd::diff_rules::{DiffOp, Add, Mul, Softmax, DiffOpRegistry};
//!
//! // Forward pass
//! let add = Add;
//! let x = vec![1.0_f64, 2.0, 3.0];
//! let y = vec![4.0_f64, 5.0, 6.0];
//! let z = add.forward(&[&x, &y]);
//! assert_eq!(z, vec![5.0, 7.0, 9.0]);
//!
//! // Backward pass (VJP)
//! let dz = vec![1.0_f64, 1.0, 1.0];
//! let grads = add.vjp(&[&x, &y], &dz);
//! assert_eq!(grads[0], vec![1.0, 1.0, 1.0]); // dL/dx
//! assert_eq!(grads[1], vec![1.0, 1.0, 1.0]); // dL/dy
//!
//! // Registry
//! let reg = DiffOpRegistry::with_standard_ops();
//! assert!(reg.get("add").is_some());
//! assert!(reg.get("softmax").is_some());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// DiffOp trait
// ─────────────────────────────────────────────────────────────────────────────

/// A differentiable operation that knows its own forward pass and VJP.
///
/// Implementors must be `Send + Sync` so they can be stored in the global
/// [`DiffOpRegistry`] and used from multiple threads.
pub trait DiffOp: Send + Sync {
    /// A stable, lowercase name used as the registry key (e.g. `"relu"`).
    fn name(&self) -> &str;

    /// Forward computation.
    ///
    /// `inputs` is a slice of input vectors.  The operation may have one or
    /// more inputs; the slice length matches the operation's arity.
    ///
    /// # Panics
    /// Implementations should not panic on reasonable inputs; they should
    /// return sensible defaults (e.g. empty `Vec`) if inputs are empty.
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64>;

    /// Vector-Jacobian product (reverse-mode gradient).
    ///
    /// Given the upstream gradient `output_grad` (same length as the forward
    /// output) and the primal `inputs`, compute and return one gradient vector
    /// per input.
    ///
    /// `output_grad.len()` equals `self.forward(inputs).len()`.
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper – element-wise combination
// ─────────────────────────────────────────────────────────────────────────────

fn elementwise_binary<F>(a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    a.iter().zip(b.iter()).map(|(&ai, &bi)| f(ai, bi)).collect()
}

fn elementwise_unary<F>(a: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    a.iter().map(|&v| f(v)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise binary ops
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise addition: z = x + y.
pub struct Add;
impl DiffOp for Add {
    fn name(&self) -> &str {
        "add"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.len() < 2 || inputs[0].is_empty() {
            return Vec::new();
        }
        elementwise_binary(inputs[0], inputs[1], |a, b| a + b)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        let n = if inputs.is_empty() { 0 } else { inputs[0].len() };
        if n == 0 {
            return vec![Vec::new(), Vec::new()];
        }
        // dL/dx = dL/dz, dL/dy = dL/dz
        vec![output_grad.to_vec(), output_grad.to_vec()]
    }
}

/// Element-wise subtraction: z = x − y.
pub struct Sub;
impl DiffOp for Sub {
    fn name(&self) -> &str {
        "sub"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.len() < 2 || inputs[0].is_empty() {
            return Vec::new();
        }
        elementwise_binary(inputs[0], inputs[1], |a, b| a - b)
    }
    fn vjp(&self, _inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        // dL/dx = dL/dz, dL/dy = -dL/dz
        let neg: Vec<f64> = output_grad.iter().map(|&v| -v).collect();
        vec![output_grad.to_vec(), neg]
    }
}

/// Element-wise multiplication: z = x ⊙ y.
pub struct Mul;
impl DiffOp for Mul {
    fn name(&self) -> &str {
        "mul"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.len() < 2 || inputs[0].is_empty() {
            return Vec::new();
        }
        elementwise_binary(inputs[0], inputs[1], |a, b| a * b)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.len() < 2 {
            return vec![Vec::new(), Vec::new()];
        }
        // dL/dx = dL/dz ⊙ y
        let dx = elementwise_binary(output_grad, inputs[1], |g, y| g * y);
        // dL/dy = dL/dz ⊙ x
        let dy = elementwise_binary(output_grad, inputs[0], |g, x| g * x);
        vec![dx, dy]
    }
}

/// Element-wise division: z = x / y.
pub struct Div;
impl DiffOp for Div {
    fn name(&self) -> &str {
        "div"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.len() < 2 || inputs[0].is_empty() {
            return Vec::new();
        }
        elementwise_binary(inputs[0], inputs[1], |a, b| a / b)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.len() < 2 {
            return vec![Vec::new(), Vec::new()];
        }
        let x = inputs[0];
        let y = inputs[1];
        // dL/dx = dL/dz / y
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(y.iter())
            .map(|(&g, &yi)| g / yi)
            .collect();
        // dL/dy = -dL/dz * x / y^2
        let dy: Vec<f64> = output_grad
            .iter()
            .zip(x.iter())
            .zip(y.iter())
            .map(|((&g, &xi), &yi)| -g * xi / (yi * yi))
            .collect();
        vec![dx, dy]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise unary ops
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise negation: z = −x.
pub struct Neg;
impl DiffOp for Neg {
    fn name(&self) -> &str {
        "neg"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], |v| -v)
    }
    fn vjp(&self, _inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        vec![elementwise_unary(output_grad, |g| -g)]
    }
}

/// Element-wise exponential: z = exp(x).
pub struct Exp;
impl DiffOp for Exp {
    fn name(&self) -> &str {
        "exp"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], f64::exp)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz * exp(x)
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| g * x.exp())
            .collect();
        vec![dx]
    }
}

/// Element-wise natural logarithm: z = log(x).
pub struct Log;
impl DiffOp for Log {
    fn name(&self) -> &str {
        "log"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], f64::ln)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz / x
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| g / x)
            .collect();
        vec![dx]
    }
}

/// Element-wise square root: z = √x.
pub struct Sqrt;
impl DiffOp for Sqrt {
    fn name(&self) -> &str {
        "sqrt"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], f64::sqrt)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz / (2 * sqrt(x))
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| g / (2.0 * x.sqrt()))
            .collect();
        vec![dx]
    }
}

/// Element-wise ReLU: z = max(0, x).
pub struct Relu;
impl DiffOp for Relu {
    fn name(&self) -> &str {
        "relu"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], |v| v.max(0.0))
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz * 𝟙(x > 0)
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect();
        vec![dx]
    }
}

/// Element-wise sigmoid: z = 1 / (1 + exp(−x)).
pub struct Sigmoid;
impl DiffOp for Sigmoid {
    fn name(&self) -> &str {
        "sigmoid"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], |x| 1.0 / (1.0 + (-x).exp()))
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz * z * (1 - z)
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| {
                let z = 1.0 / (1.0 + (-x).exp());
                g * z * (1.0 - z)
            })
            .collect();
        vec![dx]
    }
}

/// Element-wise hyperbolic tangent: z = tanh(x).
pub struct Tanh;
impl DiffOp for Tanh {
    fn name(&self) -> &str {
        "tanh"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        elementwise_unary(inputs[0], f64::tanh)
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // dL/dx = dL/dz * (1 - tanh²(x))
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| {
                let t = x.tanh();
                g * (1.0 - t * t)
            })
            .collect();
        vec![dx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Power
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise power: z = x^n.
pub struct Pow {
    /// The exponent.
    pub n: f64,
}
impl DiffOp for Pow {
    fn name(&self) -> &str {
        "pow"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let n = self.n;
        elementwise_unary(inputs[0], |x| x.powf(n))
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        let n = self.n;
        // dL/dx = n * x^(n-1) * dL/dz
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(inputs[0].iter())
            .map(|(&g, &x)| g * n * x.powf(n - 1.0))
            .collect();
        vec![dx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reduction ops
// ─────────────────────────────────────────────────────────────────────────────

/// Sum all elements: z = Σᵢ xᵢ (scalar output).
pub struct Sum;
impl DiffOp for Sum {
    fn name(&self) -> &str {
        "sum"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return vec![0.0];
        }
        vec![inputs[0].iter().sum()]
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        let g = output_grad.first().copied().unwrap_or(0.0);
        // dL/dx_i = dL/dz for all i (broadcast)
        vec![vec![g; inputs[0].len()]]
    }
}

/// Scalar multiplication: z = α · x.
pub struct Scale {
    /// Scaling factor α.
    pub alpha: f64,
}
impl DiffOp for Scale {
    fn name(&self) -> &str {
        "scale"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let a = self.alpha;
        elementwise_unary(inputs[0], |v| a * v)
    }
    fn vjp(&self, _inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        let a = self.alpha;
        vec![elementwise_unary(output_grad, |g| a * g)]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax: z_i = exp(x_i) / Σⱼ exp(x_j).
///
/// # VJP
///
/// The full Jacobian of softmax is `Jᵢⱼ = zᵢ(δᵢⱼ − zⱼ)`.
/// Given upstream gradient `g`, the VJP is:
/// `dL/dxᵢ = zᵢ · (gᵢ − Σⱼ gⱼ zⱼ)`
pub struct Softmax;
impl DiffOp for Softmax {
    fn name(&self) -> &str {
        "softmax"
    }
    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let x = inputs[0];
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }
    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.is_empty() {
            return vec![Vec::new()];
        }
        // Compute z = softmax(x) via forward pass
        let x = inputs[0];
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let z: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

        // dot = Σⱼ gⱼ * zⱼ
        let dot: f64 = output_grad.iter().zip(z.iter()).map(|(&g, &zi)| g * zi).sum();

        // dL/dxᵢ = zᵢ * (gᵢ − dot)
        let dx: Vec<f64> = output_grad
            .iter()
            .zip(z.iter())
            .map(|(&g, &zi)| zi * (g - dot))
            .collect();
        vec![dx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix multiply
// ─────────────────────────────────────────────────────────────────────────────

/// Dense matrix multiplication: Z = A @ B.
///
/// Inputs are expected as flat, row-major arrays:
/// - `inputs[0]` has length `m * k` (matrix A, shape `[m, k]`)
/// - `inputs[1]` has length `k * n` (matrix B, shape `[k, n]`)
///
/// Output has length `m * n` (matrix Z, shape `[m, n]`).
///
/// # VJP
///
/// - `dL/dA = dL/dZ @ Bᵀ`  — shape `[m, k]`
/// - `dL/dB = Aᵀ @ dL/dZ`  — shape `[k, n]`
pub struct MatMul {
    /// Rows of A (and Z).
    pub m: usize,
    /// Columns of A = Rows of B.
    pub n: usize,
    /// Columns of B (and Z).
    pub k: usize,
}

impl MatMul {
    /// Multiply two flat, row-major matrices.
    fn matmul(a: &[f64], b: &[f64], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; rows_a * cols_b];
        for i in 0..rows_a {
            for l in 0..cols_a {
                let a_il = a[i * cols_a + l];
                for j in 0..cols_b {
                    out[i * cols_b + j] += a_il * b[l * cols_b + j];
                }
            }
        }
        out
    }
}

impl DiffOp for MatMul {
    fn name(&self) -> &str {
        "matmul"
    }

    fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
        if inputs.len() < 2 {
            return Vec::new();
        }
        // A: [m, n],  B: [n, k]  => Z: [m, k]
        Self::matmul(inputs[0], inputs[1], self.m, self.n, self.k)
    }

    fn vjp(&self, inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
        if inputs.len() < 2 {
            return vec![Vec::new(), Vec::new()];
        }
        let a = inputs[0]; // [m, n]
        let b = inputs[1]; // [n, k]
        let dz = output_grad; // [m, k]

        let m = self.m;
        let n = self.n; // inner dim
        let k = self.k;

        // dL/dA = dL/dZ @ Bᵀ:  [m, k] @ [k, n] = [m, n]
        // B^T[j, l] = B[l, j]
        let mut da = vec![0.0_f64; m * n];
        for i in 0..m {
            for l in 0..n {
                let mut s = 0.0;
                for j in 0..k {
                    s += dz[i * k + j] * b[l * k + j]; // B^T[j,l] = B[l,j]
                }
                da[i * n + l] = s;
            }
        }

        // dL/dB = Aᵀ @ dL/dZ:  [n, m] @ [m, k] = [n, k]
        // A^T[l, i] = A[i, l]
        let mut db = vec![0.0_f64; n * k];
        for l in 0..n {
            for j in 0..k {
                let mut s = 0.0;
                for i in 0..m {
                    s += a[i * n + l] * dz[i * k + j];
                }
                db[l * k + j] = s;
            }
        }

        vec![da, db]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry mapping operation names to their [`DiffOp`] implementations.
///
/// # Example
/// ```rust
/// use scirs2_autograd::diff_rules::{DiffOp, DiffOpRegistry, Relu};
///
/// let mut reg = DiffOpRegistry::with_standard_ops();
/// // Look up a built-in op
/// let relu = reg.get("relu").expect("relu should be registered");
/// let z = relu.forward(&[&[-1.0, 0.5, 2.0]]);
/// assert_eq!(z, vec![0.0, 0.5, 2.0]);
///
/// // Register a custom op
/// struct MyOp;
/// impl DiffOp for MyOp {
///     fn name(&self) -> &str { "my_op" }
///     fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
///         inputs[0].iter().map(|&v| v * 2.0).collect()
///     }
///     fn vjp(&self, _inputs: &[&[f64]], output_grad: &[f64]) -> Vec<Vec<f64>> {
///         vec![output_grad.iter().map(|&g| g * 2.0).collect()]
///     }
/// }
///
/// reg.register(MyOp);
/// assert!(reg.get("my_op").is_some());
/// ```
pub struct DiffOpRegistry {
    ops: HashMap<String, Box<dyn DiffOp>>,
}

impl Default for DiffOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffOpRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all standard differentiable operations.
    pub fn with_standard_ops() -> Self {
        let mut reg = Self::new();
        reg.register(Add);
        reg.register(Sub);
        reg.register(Mul);
        reg.register(Div);
        reg.register(Neg);
        reg.register(Exp);
        reg.register(Log);
        reg.register(Sqrt);
        reg.register(Relu);
        reg.register(Sigmoid);
        reg.register(Tanh);
        reg.register(Softmax);
        reg.register(Sum);
        // Note: Scale, Pow, and MatMul have free parameters so they are not
        // pre-registered here (use register() directly with a configured instance).
        reg
    }

    /// Register a new operation.  Existing entries with the same name are replaced.
    pub fn register(&mut self, op: impl DiffOp + 'static) {
        self.ops.insert(op.name().to_string(), Box::new(op));
    }

    /// Look up a registered operation by name.
    ///
    /// Returns `None` if no operation with that name has been registered.
    pub fn get(&self, name: &str) -> Option<&dyn DiffOp> {
        self.ops.get(name).map(|b| b.as_ref())
    }

    /// Returns `true` if an operation with `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.ops.contains_key(name)
    }

    /// Iterator over all registered operation names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.ops.keys().map(|s| s.as_str())
    }

    /// Number of registered operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Add ───────────────────────────────────────────────────────────────

    #[test]
    fn test_add_forward() {
        let op = Add;
        let x = vec![1.0_f64, 2.0, 3.0];
        let y = vec![4.0_f64, 5.0, 6.0];
        let z = op.forward(&[&x, &y]);
        assert_eq!(z, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_vjp() {
        let op = Add;
        let x = vec![1.0_f64, 2.0];
        let y = vec![3.0_f64, 4.0];
        let dz = vec![1.0_f64, 2.0];
        let grads = op.vjp(&[&x, &y], &dz);
        assert_eq!(grads[0], dz);
        assert_eq!(grads[1], dz);
    }

    // ── Sub ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sub_forward() {
        let op = Sub;
        let z = op.forward(&[&[5.0_f64, 3.0], &[2.0_f64, 1.0]]);
        assert_eq!(z, vec![3.0, 2.0]);
    }

    #[test]
    fn test_sub_vjp() {
        let op = Sub;
        let dz = vec![1.0_f64, 1.0];
        let grads = op.vjp(&[&[0.0_f64], &[0.0_f64]], &dz);
        assert_eq!(grads[0], dz);
        assert_eq!(grads[1], vec![-1.0, -1.0]);
    }

    // ── Mul ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mul_forward() {
        let op = Mul;
        let z = op.forward(&[&[2.0_f64, 3.0], &[4.0_f64, 5.0]]);
        assert_eq!(z, vec![8.0, 15.0]);
    }

    #[test]
    fn test_mul_vjp() {
        let op = Mul;
        let x = vec![2.0_f64, 3.0];
        let y = vec![4.0_f64, 5.0];
        let dz = vec![1.0_f64, 1.0];
        let grads = op.vjp(&[&x, &y], &dz);
        // dL/dx = dL/dz * y, dL/dy = dL/dz * x
        assert_eq!(grads[0], vec![4.0, 5.0]);
        assert_eq!(grads[1], vec![2.0, 3.0]);
    }

    // ── Div ───────────────────────────────────────────────────────────────

    #[test]
    fn test_div_forward() {
        let op = Div;
        let z = op.forward(&[&[6.0_f64, 9.0], &[2.0_f64, 3.0]]);
        assert_eq!(z, vec![3.0, 3.0]);
    }

    #[test]
    fn test_div_vjp_numerically() {
        // Verify dL/dx and dL/dy via finite differences
        let x = vec![6.0_f64];
        let y = vec![2.0_f64];
        let op = Div;
        let dz = vec![1.0_f64];
        let grads = op.vjp(&[&x, &y], &dz);
        // dL/dx = 1/y = 0.5
        assert!((grads[0][0] - 0.5).abs() < 1e-10);
        // dL/dy = -x/y^2 = -6/4 = -1.5
        assert!((grads[1][0] + 1.5).abs() < 1e-10);
    }

    // ── Neg ───────────────────────────────────────────────────────────────

    #[test]
    fn test_neg_forward_and_vjp() {
        let op = Neg;
        let x = vec![1.0_f64, -2.0, 3.0];
        let z = op.forward(&[&x]);
        assert_eq!(z, vec![-1.0, 2.0, -3.0]);
        let grads = op.vjp(&[&x], &[1.0, 1.0, 1.0]);
        assert_eq!(grads[0], vec![-1.0, -1.0, -1.0]);
    }

    // ── Exp ───────────────────────────────────────────────────────────────

    #[test]
    fn test_exp_forward() {
        let op = Exp;
        let z = op.forward(&[&[0.0_f64, 1.0]]);
        assert!((z[0] - 1.0).abs() < 1e-10);
        assert!((z[1] - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_exp_vjp() {
        let op = Exp;
        let x = vec![1.0_f64];
        let dz = vec![1.0_f64];
        let grads = op.vjp(&[&x], &dz);
        // dL/dx = exp(x) = e
        assert!((grads[0][0] - 1.0_f64.exp()).abs() < 1e-10);
    }

    // ── Log ───────────────────────────────────────────────────────────────

    #[test]
    fn test_log_forward() {
        let op = Log;
        let z = op.forward(&[&[1.0_f64, std::f64::consts::E]]);
        assert!((z[0] - 0.0).abs() < 1e-10);
        assert!((z[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_vjp() {
        let op = Log;
        let x = vec![2.0_f64];
        let grads = op.vjp(&[&x], &[1.0]);
        assert!((grads[0][0] - 0.5).abs() < 1e-10);
    }

    // ── Sqrt ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sqrt_forward_and_vjp() {
        let op = Sqrt;
        let x = vec![4.0_f64];
        let z = op.forward(&[&x]);
        assert!((z[0] - 2.0).abs() < 1e-10);
        let grads = op.vjp(&[&x], &[1.0]);
        // dL/dx = 1 / (2*sqrt(4)) = 0.25
        assert!((grads[0][0] - 0.25).abs() < 1e-10);
    }

    // ── Relu ──────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_forward() {
        let op = Relu;
        let x = vec![-1.0_f64, 0.0, 2.0];
        let z = op.forward(&[&x]);
        assert_eq!(z, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_relu_vjp() {
        let op = Relu;
        let x = vec![-1.0_f64, 0.0, 2.0];
        let dz = vec![1.0_f64, 1.0, 1.0];
        let grads = op.vjp(&[&x], &dz);
        assert_eq!(grads[0], vec![0.0, 0.0, 1.0]);
    }

    // ── Sigmoid ───────────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_forward() {
        let op = Sigmoid;
        let z = op.forward(&[&[0.0_f64]]);
        assert!((z[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_vjp() {
        let op = Sigmoid;
        let x = vec![0.0_f64]; // sigma(0) = 0.5
        let grads = op.vjp(&[&x], &[1.0]);
        // dL/dx = sigma * (1-sigma) = 0.25
        assert!((grads[0][0] - 0.25).abs() < 1e-10);
    }

    // ── Tanh ──────────────────────────────────────────────────────────────

    #[test]
    fn test_tanh_forward_and_vjp() {
        let op = Tanh;
        let x = vec![0.0_f64];
        let z = op.forward(&[&x]);
        assert!((z[0] - 0.0).abs() < 1e-10);
        let grads = op.vjp(&[&x], &[1.0]);
        // dL/dx = 1 - tanh²(0) = 1
        assert!((grads[0][0] - 1.0).abs() < 1e-10);
    }

    // ── Pow ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pow_forward_and_vjp() {
        let op = Pow { n: 3.0 };
        let x = vec![2.0_f64];
        let z = op.forward(&[&x]);
        assert!((z[0] - 8.0).abs() < 1e-10);
        let grads = op.vjp(&[&x], &[1.0]);
        // d/dx x^3 = 3x^2 = 12
        assert!((grads[0][0] - 12.0).abs() < 1e-10);
    }

    // ── Sum ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_forward_and_vjp() {
        let op = Sum;
        let x = vec![1.0_f64, 2.0, 3.0];
        let z = op.forward(&[&x]);
        assert!((z[0] - 6.0).abs() < 1e-10);
        let grads = op.vjp(&[&x], &[2.0]);
        assert_eq!(grads[0], vec![2.0, 2.0, 2.0]);
    }

    // ── Scale ─────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_forward_and_vjp() {
        let op = Scale { alpha: 3.0 };
        let x = vec![1.0_f64, 2.0];
        let z = op.forward(&[&x]);
        assert_eq!(z, vec![3.0, 6.0]);
        let grads = op.vjp(&[&x], &[1.0, 1.0]);
        assert_eq!(grads[0], vec![3.0, 3.0]);
    }

    // ── Softmax ───────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_forward_sums_to_one() {
        let op = Softmax;
        let x = vec![1.0_f64, 2.0, 3.0];
        let z = op.forward(&[&x]);
        let s: f64 = z.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "sum={}", s);
    }

    #[test]
    fn test_softmax_vjp_numerically() {
        // Compare softmax VJP against finite difference on L = g · softmax(x)
        let op = Softmax;
        let x = vec![0.5_f64, 1.0, 0.2];
        let dz = vec![1.0_f64, 0.0, 0.0]; // L = z[0]
        let grads = op.vjp(&[&x], &dz);

        let eps = 1e-6;
        for k in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += eps;
            xm[k] -= eps;
            let zp = op.forward(&[&xp]);
            let zm = op.forward(&[&xm]);
            // L = z[0], so dL/dxk ≈ (z[0](x+ek) - z[0](x-ek)) / 2h
            let fd = (zp[0] - zm[0]) / (2.0 * eps);
            assert!(
                (grads[0][k] - fd).abs() < 1e-5,
                "k={}: vjp={} fd={}",
                k,
                grads[0][k],
                fd
            );
        }
    }

    // ── MatMul ────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_forward() {
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  =>  Z = [[19,22],[43,50]]
        let op = MatMul { m: 2, n: 2, k: 2 };
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let z = op.forward(&[&a, &b]);
        assert!((z[0] - 19.0).abs() < 1e-10);
        assert!((z[1] - 22.0).abs() < 1e-10);
        assert!((z[2] - 43.0).abs() < 1e-10);
        assert!((z[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_vjp_numerically() {
        // A = [m=2, n=3], B = [n=3, k=2] => Z = [m=2, k=2]
        let op = MatMul { m: 2, n: 3, k: 2 };
        let a: Vec<f64> = (0..6).map(|v| v as f64 + 1.0).collect(); // [1..6]
        let b: Vec<f64> = (0..6).map(|v| v as f64 + 7.0).collect(); // [7..12]
        let dz = vec![1.0_f64; 4]; // shape [m, k] = [2, 2]

        let grads = op.vjp(&[&a, &b], &dz);

        // Numerical dL/da[i,l] via central difference
        let eps = 1e-6;
        for idx in 0..6 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[idx] += eps;
            am[idx] -= eps;
            let zp = op.forward(&[&ap, &b]);
            let zm = op.forward(&[&am, &b]);
            let fd: f64 = zp.iter().zip(zm.iter()).map(|(p, m)| (p - m) / (2.0 * eps)).sum();
            assert!(
                (grads[0][idx] - fd).abs() < 1e-6,
                "da[{}]: vjp={} fd={}",
                idx,
                grads[0][idx],
                fd
            );
        }
    }

    // ── DiffOpRegistry ────────────────────────────────────────────────────

    #[test]
    fn test_registry_standard_ops() {
        let reg = DiffOpRegistry::with_standard_ops();
        for name in &[
            "add", "sub", "mul", "div", "neg", "exp", "log", "sqrt", "relu", "sigmoid", "tanh",
            "softmax", "sum",
        ] {
            assert!(reg.contains(name), "missing: {}", name);
        }
    }

    #[test]
    fn test_registry_custom_op() {
        let mut reg = DiffOpRegistry::with_standard_ops();
        struct DoubleOp;
        impl DiffOp for DoubleOp {
            fn name(&self) -> &str {
                "double"
            }
            fn forward(&self, inputs: &[&[f64]]) -> Vec<f64> {
                inputs[0].iter().map(|&v| 2.0 * v).collect()
            }
            fn vjp(&self, _inputs: &[&[f64]], og: &[f64]) -> Vec<Vec<f64>> {
                vec![og.iter().map(|&g| 2.0 * g).collect()]
            }
        }
        reg.register(DoubleOp);
        assert!(reg.contains("double"));
        let op = reg.get("double").expect("registered op should be found");
        let z = op.forward(&[&[3.0_f64, 4.0]]);
        assert_eq!(z, vec![6.0, 8.0]);
    }

    #[test]
    fn test_registry_nonexistent_returns_none() {
        let reg = DiffOpRegistry::with_standard_ops();
        assert!(reg.get("nonexistent_op_xyz").is_none());
    }
}
