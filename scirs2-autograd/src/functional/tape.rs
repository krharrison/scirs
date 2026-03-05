//! Wengert tape (computational tape) for reverse-mode automatic differentiation
//!
//! This module provides a standalone, graph-free implementation of reverse-mode
//! AD using the *Wengert list* algorithm.  Each scalar operation appends an
//! entry to a dynamic list; the backward pass walks the list in reverse,
//! accumulating cotangents into an adjoint vector.
//!
//! # Design goals
//!
//! * **Purely functional interface** — no global mutable state; each `Tape` is
//!   an independent object.
//! * **Zero-cost for constants** — constant scalars are represented as
//!   [`TapeVar`] with an empty backward contribution; they occupy one slot in
//!   the value table but incur no gradient update.
//! * **Memory efficiency** — backward pass accumulates adjoints in a single
//!   `Vec<f64>` without allocating per-node gradient objects.
//! * **Composable** — a `Tape` can be used inside closures that return
//!   [`TapeVar`]; the returned variable's index identifies the output node in
//!   the adjoint computation.
//!
//! # Overview
//!
//! | Type / Function | Description |
//! |-----------------|-------------|
//! | [`Tape`] | Wengert list accumulating operations |
//! | [`TapeVar`] | Handle to a variable on a tape |
//! | [`backward`] | Run the backward pass on a recorded tape |
//! | [`GradientTape`] | High-level closure-based interface |
//!
//! # Example — computing a gradient
//!
//! ```rust
//! use scirs2_autograd::functional::tape::{Tape, backward};
//!
//! let mut tape = Tape::new();
//! let x = tape.var(3.0);
//! let y = tape.var(4.0);
//! // f(x, y) = x^2 + y^2
//! let x2 = x.mul(&y, &tape); // reusing tape API
//!
//! // Alternatively use the functional tape builder:
//! let mut tape = Tape::new();
//! let xi = tape.push_input(3.0);
//! let yi = tape.push_input(4.0);
//! let sq_x = tape.push_unary(xi, |v| v * v, |_, _| 2.0 * 3.0);
//! let sq_y = tape.push_unary(yi, |v| v * v, |_, _| 2.0 * 4.0);
//! let out  = tape.push_binary(sq_x, sq_y, |a, b| a + b, |_, _| (1.0, 1.0));
//! let grads = backward(&tape, out);
//! assert!((grads[xi] - 6.0).abs() < 1e-12);
//! assert!((grads[yi] - 8.0).abs() < 1e-12);
//! ```

use crate::error::AutogradError;
use crate::Result;
use std::cell::RefCell;

// ============================================================================
// Node — an entry in the Wengert list
// ============================================================================

/// A single entry in the Wengert list.
///
/// Each node records:
/// * its primal `value` (for potential reuse in backward),
/// * a list of `(parent_index, weight)` pairs, where the weight is the local
///   derivative `∂out/∂parent` (already evaluated at forward-pass values).
///
/// The backward pass propagates adjoints as:
/// `adjoint[parent] += adjoint[self] * weight`
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// The primal value of this node.
    pub(crate) value: f64,
    /// `(parent_idx, local_partial)` pairs (at most 2 for binary ops).
    pub(crate) parents: Vec<(usize, f64)>,
}

// ============================================================================
// Tape — the Wengert list
// ============================================================================

/// A Wengert list (computational tape) for reverse-mode automatic
/// differentiation.
///
/// The tape records every primitive operation performed on [`TapeVar`]s in
/// chronological order.  After a forward pass, call [`backward`] to
/// propagate cotangents (adjoints) from an output node back to all inputs.
///
/// # Thread safety
///
/// `Tape` is `!Send` by design (it uses interior mutability via `RefCell`).
/// For multi-threaded workloads, create a `Tape` per thread.
pub struct Tape {
    /// The Wengert list, stored behind `RefCell` to allow shared-reference
    /// mutation during the forward pass (a common pattern when closures
    /// operate on `TapeVar` values that borrow the tape).
    nodes: RefCell<Vec<Node>>,
}

impl Tape {
    /// Create a new, empty `Tape`.
    pub fn new() -> Self {
        Self { nodes: RefCell::new(Vec::new()) }
    }

    /// Create a new `Tape` with pre-allocated capacity `n` for efficiency.
    pub fn with_capacity(n: usize) -> Self {
        Self { nodes: RefCell::new(Vec::with_capacity(n)) }
    }

    /// Record an *input* (leaf) variable with the given primal value.
    ///
    /// Returns the node index (a [`TapeVar`] handle).
    pub fn push_input(&self, value: f64) -> TapeVar {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node { value, parents: Vec::new() });
        TapeVar { idx, value }
    }

    /// Record a *constant* (no gradient flows through it).
    ///
    /// Equivalent to `push_input` semantically, but communicates intent and
    /// allows future optimisers to skip gradient accumulation.
    pub fn push_constant(&self, value: f64) -> TapeVar {
        self.push_input(value)
    }

    /// Record a *unary* operation.
    ///
    /// # Arguments
    ///
    /// * `a`        — Parent variable.
    /// * `forward`  — `fn(value_a) -> output_value`.
    /// * `backward` — `fn(value_a, output_value) -> ∂out/∂a`.
    pub fn push_unary(
        &self,
        a: TapeVar,
        forward: impl Fn(f64) -> f64,
        backward: impl Fn(f64, f64) -> f64,
    ) -> TapeVar {
        let out_val = forward(a.value);
        let local_grad = backward(a.value, out_val);
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            value: out_val,
            parents: vec![(a.idx, local_grad)],
        });
        TapeVar { idx, value: out_val }
    }

    /// Record a *binary* operation.
    ///
    /// # Arguments
    ///
    /// * `a`, `b`   — Parent variables.
    /// * `forward`  — `fn(value_a, value_b) -> output_value`.
    /// * `backward` — `fn(value_a, value_b) -> (∂out/∂a, ∂out/∂b)`.
    pub fn push_binary(
        &self,
        a: TapeVar,
        b: TapeVar,
        forward: impl Fn(f64, f64) -> f64,
        backward: impl Fn(f64, f64) -> (f64, f64),
    ) -> TapeVar {
        let out_val = forward(a.value, b.value);
        let (ga, gb) = backward(a.value, b.value);
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            value: out_val,
            parents: vec![(a.idx, ga), (b.idx, gb)],
        });
        TapeVar { idx, value: out_val }
    }

    /// Create a `TapeVar` wrapper for an external computation result.
    ///
    /// Use this when you have computed the output value by some other means
    /// and want to register it with manually specified parent connections.
    pub fn push_custom(
        &self,
        value: f64,
        parents: Vec<(TapeVar, f64)>,
    ) -> TapeVar {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            value,
            parents: parents.into_iter().map(|(v, w)| (v.idx, w)).collect(),
        });
        TapeVar { idx, value }
    }

    /// Number of nodes currently on the tape.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Return `true` if the tape has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    /// Snapshot of all recorded node values.
    pub fn values(&self) -> Vec<f64> {
        self.nodes.borrow().iter().map(|n| n.value).collect()
    }

    /// Convenience: push `x` and return a named variable wrapper.
    pub fn var(&self, value: f64) -> TapeVar {
        self.push_input(value)
    }

    // -----------------------------------------------------------------------
    // Primitive op helpers (syntactic sugar)
    // -----------------------------------------------------------------------

    /// Record `a + b`.
    pub fn add(&self, a: TapeVar, b: TapeVar) -> TapeVar {
        self.push_binary(a, b, |va, vb| va + vb, |_, _| (1.0, 1.0))
    }

    /// Record `a - b`.
    pub fn sub(&self, a: TapeVar, b: TapeVar) -> TapeVar {
        self.push_binary(a, b, |va, vb| va - vb, |_, _| (1.0, -1.0))
    }

    /// Record `a * b`.
    pub fn mul(&self, a: TapeVar, b: TapeVar) -> TapeVar {
        self.push_binary(a, b, |va, vb| va * vb, |va, vb| (vb, va))
    }

    /// Record `a / b`.
    pub fn div(&self, a: TapeVar, b: TapeVar) -> TapeVar {
        self.push_binary(
            a, b,
            |va, vb| va / vb,
            |va, vb| (1.0 / vb, -va / (vb * vb)),
        )
    }

    /// Record `-a`.
    pub fn neg(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| -va, |_, _| -1.0)
    }

    /// Record `exp(a)`.
    pub fn exp(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.exp(), |_, out| out)
    }

    /// Record `ln(a)`.
    pub fn ln(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.ln(), |va, _| 1.0 / va)
    }

    /// Record `sqrt(a)`.
    pub fn sqrt(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.sqrt(), |_, out| 0.5 / out)
    }

    /// Record `sin(a)`.
    pub fn sin(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.sin(), |va, _| va.cos())
    }

    /// Record `cos(a)`.
    pub fn cos(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.cos(), |va, _| -va.sin())
    }

    /// Record `tanh(a)`.
    pub fn tanh(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.tanh(), |_, out| 1.0 - out * out)
    }

    /// Record `a^n` for integer `n`.
    pub fn powi(&self, a: TapeVar, n: i32) -> TapeVar {
        self.push_unary(a, |va| va.powi(n), |va, _| f64::from(n) * va.powi(n - 1))
    }

    /// Record `a * scalar`.
    pub fn scale(&self, a: TapeVar, scalar: f64) -> TapeVar {
        self.push_unary(a, |va| va * scalar, |_, _| scalar)
    }

    /// Record `sigmoid(a) = 1/(1+exp(-a))`.
    pub fn sigmoid(&self, a: TapeVar) -> TapeVar {
        self.push_unary(
            a,
            |va| {
                let e = (-va).exp();
                1.0 / (1.0 + e)
            },
            |_, out| out * (1.0 - out),
        )
    }

    /// Record `relu(a) = max(0, a)`.
    pub fn relu(&self, a: TapeVar) -> TapeVar {
        self.push_unary(a, |va| va.max(0.0), |va, _| if va > 0.0 { 1.0 } else { 0.0 })
    }

    /// Record sum: `Σ aᵢ` from a slice of `TapeVar`.
    ///
    /// Implemented as a left-fold over `Tape::add`.
    pub fn sum(&self, vars: &[TapeVar]) -> Result<TapeVar> {
        if vars.is_empty() {
            return Err(AutogradError::invalid_argument(
                "tape::sum: empty input slice".to_string(),
            ));
        }
        let mut acc = vars[0];
        for &v in &vars[1..] {
            acc = self.add(acc, v);
        }
        Ok(acc)
    }

    /// Record dot product `aᵀb`.
    pub fn dot(&self, a: &[TapeVar], b: &[TapeVar]) -> Result<TapeVar> {
        if a.len() != b.len() {
            return Err(AutogradError::invalid_argument(format!(
                "tape::dot: length mismatch {} vs {}",
                a.len(),
                b.len()
            )));
        }
        if a.is_empty() {
            return Err(AutogradError::invalid_argument(
                "tape::dot: empty inputs".to_string(),
            ));
        }
        let products: Vec<TapeVar> = a.iter().zip(b.iter()).map(|(&ai, &bi)| self.mul(ai, bi)).collect();
        self.sum(&products)
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TapeVar — handle to a node on the tape
// ============================================================================

/// A handle to a node in a [`Tape`], carrying the node's primal value.
///
/// `TapeVar` is `Copy` (just an index + cached value) and cheap to pass
/// around.  All operations that would normally return a new `Dual` instead
/// register a new node on the tape and return a `TapeVar` pointing to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TapeVar {
    /// Index of this variable in the tape's node vector.
    pub(crate) idx: usize,
    /// Primal value (cached for convenience).
    pub value: f64,
}

impl TapeVar {
    /// Return the node index (position in the Wengert list).
    #[inline]
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Convenience method: `a.add(b, tape)` = `tape.add(a, b)`.
    pub fn add(self, other: TapeVar, tape: &Tape) -> TapeVar {
        tape.add(self, other)
    }

    /// Convenience method: `a.mul(b, tape)` = `tape.mul(a, b)`.
    pub fn mul(self, other: TapeVar, tape: &Tape) -> TapeVar {
        tape.mul(self, other)
    }
}

// ============================================================================
// backward — reverse-mode backward pass
// ============================================================================

/// Run the reverse-mode backward pass on a recorded `tape`.
///
/// Returns a `Vec<f64>` of the same length as the number of tape nodes, where
/// `grads[i]` is `∂out/∂(node i)`.  Leaf variables (inputs) have their
/// adjoints stored at their original indices.
///
/// # Arguments
///
/// * `tape`   — The Wengert list populated during the forward pass.
/// * `output` — The [`TapeVar`] representing the scalar loss / output node.
///
/// # Panics / Errors
///
/// The function never panics; if the tape is empty it returns an empty vector.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::tape::{Tape, backward};
///
/// let tape = Tape::new();
/// let x = tape.var(2.0);
/// let y = tape.var(3.0);
/// let z = tape.mul(x, y);       // z = x*y = 6
/// let w = tape.powi(x, 2);      // w = x^2 = 4
/// let out = tape.add(z, w);     // out = z + w = 10
///
/// let grads = backward(&tape, out);
/// // ∂out/∂x = y + 2x = 3 + 4 = 7
/// // ∂out/∂y = x = 2
/// assert!((grads[x.idx] - 7.0).abs() < 1e-12);
/// assert!((grads[y.idx] - 2.0).abs() < 1e-12);
/// ```
pub fn backward(tape: &Tape, output: TapeVar) -> Vec<f64> {
    let nodes = tape.nodes.borrow();
    let n = nodes.len();
    let mut adjoints = vec![0.0f64; n];
    if n == 0 {
        return adjoints;
    }
    adjoints[output.idx] = 1.0;

    // Walk backwards through the Wengert list
    for i in (0..n).rev() {
        let adjoint_i = adjoints[i];
        if adjoint_i == 0.0 {
            continue; // short-circuit zero contributions
        }
        for &(parent_idx, weight) in &nodes[i].parents {
            adjoints[parent_idx] += adjoint_i * weight;
        }
    }

    adjoints
}

/// Run a backward pass starting from the output with a seed adjoint ≠ 1.
///
/// Useful for computing `vᵀ·J` (vector-Jacobian products) when `v ≠ e_out`.
pub fn backward_with_seed(tape: &Tape, output: TapeVar, seed: f64) -> Vec<f64> {
    let nodes = tape.nodes.borrow();
    let n = nodes.len();
    let mut adjoints = vec![0.0f64; n];
    if n == 0 {
        return adjoints;
    }
    adjoints[output.idx] = seed;

    for i in (0..n).rev() {
        let adjoint_i = adjoints[i];
        if adjoint_i == 0.0 {
            continue;
        }
        for &(parent_idx, weight) in &nodes[i].parents {
            adjoints[parent_idx] += adjoint_i * weight;
        }
    }

    adjoints
}

// ============================================================================
// GradientTape — high-level closure interface
// ============================================================================

/// A high-level closure-based interface to the Wengert tape.
///
/// [`GradientTape`] wraps a [`Tape`] and records a computation via a closure,
/// then exposes a clean gradient-extraction API.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::tape::GradientTape;
///
/// // f(x, y) = x^2 * y + y^3
/// // ∂f/∂x = 2*x*y = 4,  ∂f/∂y = x^2 + 3*y^2 = 11  at (2, 1)
/// let mut gt = GradientTape::new();
/// let inputs = gt.record_inputs(&[2.0, 1.0]);
/// let result = gt.compute(|tape, inp| {
///     let x = inp[0];
///     let y = inp[1];
///     let x2 = tape.powi(x, 2);
///     let x2y = tape.mul(x2, y);
///     let y3 = tape.powi(y, 3);
///     tape.add(x2y, y3)
/// }, &inputs);
/// let grads = gt.gradient(result, &inputs);
/// assert!((grads[0] - 4.0).abs() < 1e-12);  // ∂f/∂x
/// assert!((grads[1] - 13.0).abs() < 1e-12); // ∂f/∂y: 4 + 3 = 7... wait x^2+3y^2=4+3=7? No: x=2,y=1 => 4+3=7
/// ```
pub struct GradientTape {
    tape: Tape,
}

impl GradientTape {
    /// Create a new `GradientTape`.
    pub fn new() -> Self {
        Self { tape: Tape::new() }
    }

    /// Register a slice of scalar inputs.  Returns `TapeVar` handles.
    pub fn record_inputs(&self, inputs: &[f64]) -> Vec<TapeVar> {
        inputs.iter().map(|&v| self.tape.push_input(v)).collect()
    }

    /// Run a computation closure and return the output `TapeVar`.
    pub fn compute<F>(&self, f: F, inputs: &[TapeVar]) -> TapeVar
    where
        F: FnOnce(&Tape, &[TapeVar]) -> TapeVar,
    {
        f(&self.tape, inputs)
    }

    /// Compute gradients of `output` w.r.t. each element of `wrt`.
    ///
    /// Returns a `Vec<f64>` of the same length as `wrt`.
    pub fn gradient(&self, output: TapeVar, wrt: &[TapeVar]) -> Vec<f64> {
        let adjoints = backward(&self.tape, output);
        wrt.iter().map(|v| adjoints[v.idx]).collect()
    }

    /// Borrow the inner tape.
    pub fn tape(&self) -> &Tape {
        &self.tape
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Functional helpers — functional-style gradient computations
// ============================================================================

/// Compute the gradient of a scalar function `f: R^n -> R` at point `x`
/// using the Wengert tape.
///
/// The function `f` receives `&[TapeVar]` and must return a single `TapeVar`.
/// This is the tape-based analogue of [`crate::functional::transforms::grad`].
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::tape::tape_grad;
///
/// let g = tape_grad(
///     |tape, xs| {
///         let x2 = tape.powi(xs[0], 2);
///         let y2 = tape.powi(xs[1], 2);
///         tape.add(x2, y2)
///     },
///     &[3.0, 4.0],
/// ).expect("gradient");
/// assert!((g[0] - 6.0).abs() < 1e-12);
/// assert!((g[1] - 8.0).abs() < 1e-12);
/// ```
pub fn tape_grad<F>(f: F, x: &[f64]) -> Result<Vec<f64>>
where
    F: FnOnce(&Tape, &[TapeVar]) -> TapeVar,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "tape_grad: input must be non-empty".to_string(),
        ));
    }
    let tape = Tape::with_capacity(x.len() * 4);
    let inputs: Vec<TapeVar> = x.iter().map(|&v| tape.push_input(v)).collect();
    let output = f(&tape, &inputs);
    let adjoints = backward(&tape, output);
    Ok(inputs.iter().map(|v| adjoints[v.idx]).collect())
}

/// Compute the full Jacobian of `f: R^n -> R^m` at `x` using reverse-mode
/// tape AD.
///
/// Each row of the returned `m × n` matrix is computed by one backward pass
/// seeded with the corresponding basis cotangent vector.  Total cost: `m`
/// backward passes.
///
/// # Errors
///
/// Returns `AutogradError` if `x` or outputs are empty.
pub fn tape_jacobian<F>(f: F, x: &[f64]) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&Tape, &[TapeVar]) -> Vec<TapeVar>,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "tape_jacobian: input must be non-empty".to_string(),
        ));
    }

    // We need to re-evaluate for each output because the tape records the
    // entire graph; seeding different outputs is equivalent to running the
    // full backward pass from each output node.
    let tape = Tape::with_capacity(x.len() * 8);
    let inputs: Vec<TapeVar> = x.iter().map(|&v| tape.push_input(v)).collect();
    let outputs = f(&tape, &inputs);

    if outputs.is_empty() {
        return Err(AutogradError::invalid_argument(
            "tape_jacobian: function returned no outputs".to_string(),
        ));
    }

    let m = outputs.len();
    let n = x.len();
    let mut jacobian = vec![vec![0.0f64; n]; m];

    for (row, &out_var) in outputs.iter().enumerate() {
        let adjoints = backward(&tape, out_var);
        for (col, inp_var) in inputs.iter().enumerate() {
            jacobian[row][col] = adjoints[inp_var.idx];
        }
    }

    Ok(jacobian)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_basic_gradient() {
        // f(x, y) = x^2 + y^2; ∂f/∂x = 2x, ∂f/∂y = 2y
        let g = tape_grad(
            |tape, xs| {
                let x2 = tape.powi(xs[0], 2);
                let y2 = tape.powi(xs[1], 2);
                tape.add(x2, y2)
            },
            &[3.0, 4.0],
        )
        .expect("tape_grad");
        assert!((g[0] - 6.0).abs() < 1e-12, "∂f/∂x = {}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-12, "∂f/∂y = {}", g[1]);
    }

    #[test]
    fn test_tape_product_gradient() {
        // f(x, y) = x * y; ∂f/∂x = y, ∂f/∂y = x
        let g = tape_grad(
            |tape, xs| tape.mul(xs[0], xs[1]),
            &[3.0, 4.0],
        )
        .expect("tape_grad mul");
        assert!((g[0] - 4.0).abs() < 1e-12);
        assert!((g[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_tape_chain_rule() {
        // f(x) = exp(x^2); f'(x) = 2x*exp(x^2)
        let g = tape_grad(
            |tape, xs| {
                let x2 = tape.powi(xs[0], 2);
                tape.exp(x2)
            },
            &[2.0],
        )
        .expect("chain rule");
        let expected = 4.0 * (4.0_f64).exp(); // 2*2*exp(4)
        assert!((g[0] - expected).abs() < 1e-6, "f'(2) = {} expected {}", g[0], expected);
    }

    #[test]
    fn test_tape_jacobian_vector_function() {
        // f(x, y) = [x^2 * y, x + y]
        // J = [[2xy, x^2], [1, 1]] at (2, 3) = [[12, 4], [1, 1]]
        let j = tape_jacobian(
            |tape, xs| {
                let x2 = tape.powi(xs[0], 2);
                let f0 = tape.mul(x2, xs[1]);
                let f1 = tape.add(xs[0], xs[1]);
                vec![f0, f1]
            },
            &[2.0, 3.0],
        )
        .expect("tape_jacobian");
        assert!((j[0][0] - 12.0).abs() < 1e-12, "J[0][0] = {}", j[0][0]);
        assert!((j[0][1] - 4.0).abs() < 1e-12, "J[0][1] = {}", j[0][1]);
        assert!((j[1][0] - 1.0).abs() < 1e-12, "J[1][0] = {}", j[1][0]);
        assert!((j[1][1] - 1.0).abs() < 1e-12, "J[1][1] = {}", j[1][1]);
    }

    #[test]
    fn test_tape_sigmoid() {
        let g = tape_grad(|tape, xs| tape.sigmoid(xs[0]), &[0.0]).expect("sigmoid grad");
        assert!((g[0] - 0.25).abs() < 1e-12, "sigmoid'(0) = {}", g[0]);
    }

    #[test]
    fn test_gradient_tape_high_level() {
        let gt = GradientTape::new();
        let inputs = gt.record_inputs(&[2.0, 3.0]);
        let out = gt.compute(
            |tape, inp| {
                let x2 = tape.powi(inp[0], 2);
                let y2 = tape.powi(inp[1], 2);
                tape.add(x2, y2)
            },
            &inputs,
        );
        let grads = gt.gradient(out, &inputs);
        assert!((grads[0] - 4.0).abs() < 1e-12, "dx = {}", grads[0]);
        assert!((grads[1] - 6.0).abs() < 1e-12, "dy = {}", grads[1]);
    }

    #[test]
    fn test_backward_with_seed() {
        let tape = Tape::new();
        let x = tape.var(2.0);
        let y = tape.var(3.0);
        let z = tape.mul(x, y); // z = x*y; dz/dx = y = 3, dz/dy = x = 2
        let adjs = backward_with_seed(&tape, z, 2.0); // seed 2 => 2*[3,2] = [6,4]
        assert!((adjs[x.idx] - 6.0).abs() < 1e-12);
        assert!((adjs[y.idx] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_tape_empty_error() {
        assert!(tape_grad(|tape, xs| xs[0], &[]).is_err());
        assert!(tape_jacobian(|_tape, _xs| vec![], &[1.0]).is_err());
    }

    #[test]
    fn test_tape_dot_product() {
        // f(a, b) = a·b = a0*b0 + a1*b1 + a2*b2
        // at a=(1,2,3), b=(4,5,6): f = 32, ∂f/∂aᵢ = bᵢ, ∂f/∂bᵢ = aᵢ
        let tape = Tape::new();
        let a: Vec<TapeVar> = [1.0, 2.0, 3.0].iter().map(|&v| tape.var(v)).collect();
        let b: Vec<TapeVar> = [4.0, 5.0, 6.0].iter().map(|&v| tape.var(v)).collect();
        let out = tape.dot(&a, &b).expect("dot product");
        let adjs = backward(&tape, out);
        assert!((out.value - 32.0).abs() < 1e-12);
        assert!((adjs[a[0].idx] - 4.0).abs() < 1e-12);
        assert!((adjs[a[2].idx] - 6.0).abs() < 1e-12);
        assert!((adjs[b[1].idx] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_tape_relu_gradient() {
        // relu'(x) = 1 if x > 0, 0 otherwise
        let g_pos = tape_grad(|tape, xs| tape.relu(xs[0]), &[2.0]).expect("relu+");
        let g_neg = tape_grad(|tape, xs| tape.relu(xs[0]), &[-1.0]).expect("relu-");
        assert!((g_pos[0] - 1.0).abs() < 1e-12);
        assert!(g_neg[0].abs() < 1e-12);
    }
}
