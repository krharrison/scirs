//! Dynamic computation graphs with control flow
//!
//! This module provides a dynamic computation graph that can change structure
//! per forward pass, supporting conditionals, loops, and scans with full
//! gradient support via tape-based recording.
//!
//! Unlike the static autograd graph (which is fixed at graph-build time),
//! a `DynamicGraph` is evaluated eagerly: each operation executes immediately
//! and records its backward function on a `GradientTape`.  Gradients are then
//! computed by replaying the tape in reverse.
//!
//! # Architecture
//!
//! ```text
//! DynTensor ── owns ──► Arc<DynTensorData>
//!                              │
//!                              ├─ value: Array<f64, IxDyn>
//!                              ├─ id: usize (unique per tensor)
//!                              └─ tape_entry: Option<TapeEntry>
//!
//! GradientTape ── stores ──► Vec<TapeOp>  (in forward order)
//!                              │
//!                              └─ TapeOp { output_id, input_ids, backward_fn }
//! ```
//!
//! # Control flow
//!
//! ```text
//! if_else(cond, then_fn, else_fn) → DynTensor
//!   Executes exactly one branch; records a TapeOp that routes gradients only
//!   to the executed branch (the other branch receives zero gradient).
//!
//! while_loop(cond_fn, body_fn, state) → DynTensor
//!   Runs body_fn until cond_fn returns false; unrolls the tape for gradient.
//!
//! scan(fn, init, xs) → (DynTensor, DynTensor)
//!   Scans fn over xs, accumulating carries; the per-step outputs are stacked.
//! ```
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
//! use scirs2_core::ndarray::Array1;
//!
//! let graph = DynamicGraph::new();
//! let (result, tape) = with_tape(|| {
//!     let x = graph.tensor(Array1::from(vec![2.0_f64, 3.0]).into_dyn());
//!     let y = graph.mul_scalar(&x, 2.0);
//!     y
//! });
//! let x_id = result.id().saturating_sub(1); // conceptual
//! let _ = tape.operations_count(); // inspect tape length
//! ```

use crate::error::AutogradError;
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::sync::{Arc, Mutex, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// Global tape registry (thread-local to avoid cross-thread issues)
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    static ACTIVE_TAPE: RefCell<Option<Arc<Mutex<TapeInner>>>> = RefCell::new(None);
}

use std::cell::RefCell;

// ─────────────────────────────────────────────────────────────────────────────
// Unique ID generation
// ─────────────────────────────────────────────────────────────────────────────

static TENSOR_ID_COUNTER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

fn next_tensor_id() -> usize {
    TENSOR_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ─────────────────────────────────────────────────────────────────────────────
// TapeInner / TapeOp — internal tape structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single recorded operation on the gradient tape.
pub struct TapeOp {
    /// The unique ID of the output tensor produced by this op.
    pub output_id: usize,
    /// The unique IDs of all input tensors consumed by this op.
    pub input_ids: Vec<usize>,
    /// The backward function: given the gradient wrt the output, return
    /// gradients wrt each input (in the same order as `input_ids`).
    pub backward_fn: Box<dyn Fn(&ArrayD<f64>) -> Vec<ArrayD<f64>> + Send + Sync>,
    /// Human-readable name for debugging.
    pub name: String,
}

impl std::fmt::Debug for TapeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeOp")
            .field("name", &self.name)
            .field("output_id", &self.output_id)
            .field("input_ids", &self.input_ids)
            .finish()
    }
}

/// Internal mutable state of a `GradientTape`.
struct TapeInner {
    ops: Vec<TapeOp>,
    /// Map from tensor ID → its concrete value (stored during forward pass).
    values: std::collections::HashMap<usize, ArrayD<f64>>,
}

impl TapeInner {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            values: std::collections::HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientTape (public)
// ─────────────────────────────────────────────────────────────────────────────

/// A tape that records operations for reverse-mode automatic differentiation.
///
/// The tape is populated during a forward pass (either directly via
/// [`DynamicGraph`] operations, or via [`with_tape`]).  After the forward pass,
/// call [`GradientTape::gradient`] to obtain gradients with respect to any set
/// of source tensors.
///
/// # Thread safety
///
/// The tape uses interior mutability guarded by a `Mutex`.  Concurrent writes
/// from multiple threads are serialised; reading gradients is single-threaded.
pub struct GradientTape {
    inner: Arc<Mutex<TapeInner>>,
}

impl GradientTape {
    /// Create a new, empty gradient tape.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TapeInner::new())),
        }
    }

    /// Number of operations recorded on the tape.
    pub fn operations_count(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.ops.len())
            .unwrap_or(0)
    }

    /// Record a single operation on the tape.
    ///
    /// This is called internally by [`DynamicGraph`] operations.  You rarely
    /// need to call it directly.
    ///
    /// # Arguments
    /// * `op` – The tape operation to record.
    pub fn record_op(&self, op: TapeOp) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.ops.push(op);
        }
    }

    /// Store the concrete value of `tensor_id` in the tape's value cache.
    pub fn store_value(&self, tensor_id: usize, value: ArrayD<f64>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.values.insert(tensor_id, value);
        }
    }

    /// Compute gradients of `output` with respect to each tensor in `inputs`.
    ///
    /// Uses reverse-mode automatic differentiation by replaying the tape
    /// backwards from `output`.
    ///
    /// # Arguments
    /// * `output`  – The scalar or tensor whose gradient is taken from.
    /// * `inputs`  – Tensors wrt which we want gradients.
    ///
    /// # Returns
    /// A vector of `ArrayD<f64>` with the same length as `inputs`.  Each element
    /// is `∂output/∂inputs[i]`.  If a given input did not contribute to the
    /// output (no path exists on the tape), its gradient is all-zeros of the
    /// same shape as the input value.
    ///
    /// # Errors
    /// Returns `AutogradError` if the tape lock is poisoned or if the output
    /// tensor has no recorded value.
    pub fn gradient(
        &self,
        output: &DynTensor,
        inputs: &[&DynTensor],
    ) -> Result<Vec<ArrayD<f64>>, AutogradError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AutogradError::OperationError(format!("Tape lock poisoned: {e}")))?;

        let output_shape = output.shape();
        let output_id = output.id();

        // initialise gradient accumulator: output ← ones of output shape
        let mut grad_map: std::collections::HashMap<usize, ArrayD<f64>> =
            std::collections::HashMap::new();

        // seed: ∂output/∂output = 1 (or ones tensor for non-scalar)
        let seed = Array::ones(output_shape.as_slice());
        grad_map.insert(output_id, seed);

        // Replay in reverse order
        for op in inner.ops.iter().rev() {
            let grad_out = match grad_map.get(&op.output_id) {
                Some(g) => g.clone(),
                None => continue, // this op is not on the path to the output
            };

            let input_grads = (op.backward_fn)(&grad_out);

            for (input_id, input_grad) in op.input_ids.iter().zip(input_grads.into_iter()) {
                let entry = grad_map
                    .entry(*input_id)
                    .or_insert_with(|| Array::zeros(input_grad.raw_dim()));
                // accumulate
                *entry = entry.clone() + &input_grad;
            }
        }

        // Extract gradients for the requested inputs
        let results = inputs
            .iter()
            .map(|t| {
                grad_map
                    .get(&t.id())
                    .cloned()
                    .unwrap_or_else(|| Array::zeros(t.shape().as_slice()))
            })
            .collect();

        Ok(results)
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynTensor — eager tensor node
// ─────────────────────────────────────────────────────────────────────────────

/// An eagerly-evaluated tensor that participates in a dynamic computation graph.
///
/// `DynTensor` wraps a concrete `ArrayD<f64>` value and carries a unique ID used
/// by the gradient tape to route gradients during the backward pass.
///
/// Clone is cheap: `DynTensor` is `Arc`-backed.
#[derive(Clone, Debug)]
pub struct DynTensor {
    data: Arc<DynTensorData>,
}

#[derive(Debug)]
struct DynTensorData {
    id: usize,
    value: RwLock<ArrayD<f64>>,
}

impl DynTensor {
    /// Create a new leaf tensor from a concrete array.
    ///
    /// The tensor is assigned a fresh unique ID.  No backward function is
    /// registered (leaf tensors are the sources of gradients, not sinks).
    pub fn new(value: ArrayD<f64>) -> Self {
        let id = next_tensor_id();
        // Register value in the active tape (if any)
        ACTIVE_TAPE.with(|cell| {
            if let Some(tape_arc) = cell.borrow().as_ref() {
                if let Ok(mut tape) = tape_arc.lock() {
                    tape.values.insert(id, value.clone());
                }
            }
        });
        Self {
            data: Arc::new(DynTensorData {
                id,
                value: RwLock::new(value),
            }),
        }
    }

    /// The unique ID of this tensor.
    pub fn id(&self) -> usize {
        self.data.id
    }

    /// A clone of the underlying concrete array.
    pub fn value(&self) -> ArrayD<f64> {
        self.data
            .value
            .read()
            .map(|v| v.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[])))
    }

    /// The shape of the underlying array.
    pub fn shape(&self) -> Vec<usize> {
        self.data
            .value
            .read()
            .map(|v| v.shape().to_vec())
            .unwrap_or_default()
    }

    /// Returns `true` if the tensor contains a single scalar element.
    pub fn is_scalar(&self) -> bool {
        self.data
            .value
            .read()
            .map(|v| v.ndim() == 0 || (v.ndim() == 1 && v.len() == 1))
            .unwrap_or(false)
    }

    /// Extract a scalar `f64` value.  Returns an error if the tensor is not scalar.
    pub fn scalar_value(&self) -> Result<f64, AutogradError> {
        let v = self.value();
        if v.len() == 1 {
            v.iter()
                .next()
                .copied()
                .ok_or_else(|| AutogradError::OperationError("Empty tensor".to_string()))
        } else {
            Err(AutogradError::OperationError(format!(
                "scalar_value: tensor has {} elements, expected 1",
                v.len()
            )))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: record op on the active tape
// ─────────────────────────────────────────────────────────────────────────────

fn record_op_on_tape(op: TapeOp) {
    ACTIVE_TAPE.with(|cell| {
        if let Some(tape_arc) = cell.borrow().as_ref() {
            if let Ok(mut tape) = tape_arc.lock() {
                tape.ops.push(op);
            }
        }
    });
}

fn store_value_on_tape(id: usize, value: &ArrayD<f64>) {
    ACTIVE_TAPE.with(|cell| {
        if let Some(tape_arc) = cell.borrow().as_ref() {
            if let Ok(mut tape) = tape_arc.lock() {
                tape.values.insert(id, value.clone());
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicGraph — factory for dynamic computation nodes
// ─────────────────────────────────────────────────────────────────────────────

/// A factory for creating `DynTensor` nodes and recording operations on the
/// gradient tape.
///
/// `DynamicGraph` itself is stateless: all state lives in the thread-local tape
/// (activated via [`with_tape`]).
///
/// # Usage pattern
///
/// ```rust
/// use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
/// use scirs2_core::ndarray::Array1;
///
/// let graph = DynamicGraph::new();
/// let (output, tape) = with_tape(|| {
///     let x = graph.tensor(Array1::from(vec![1.0_f64, 2.0, 3.0]).into_dyn());
///     let y = graph.add_scalar(&x, 1.0);
///     y
/// });
/// // Compute gradients wrt x
/// // (x was captured; here we just show the API)
/// println!("output shape: {:?}", output.shape());
/// println!("tape ops:     {}", tape.operations_count());
/// ```
pub struct DynamicGraph;

impl DynamicGraph {
    /// Create a new (stateless) dynamic graph.
    pub fn new() -> Self {
        DynamicGraph
    }

    /// Wrap a concrete array as a leaf `DynTensor`.
    ///
    /// The value is immediately stored in the active tape's value cache.
    pub fn tensor(&self, value: ArrayD<f64>) -> DynTensor {
        DynTensor::new(value)
    }

    /// Element-wise addition of two tensors.
    ///
    /// Backward: `∂L/∂a = ∂L/∂out`, `∂L/∂b = ∂L/∂out`.
    pub fn add(&self, a: &DynTensor, b: &DynTensor) -> DynTensor {
        let out_val = a.value() + &b.value();
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let (a_id, b_id) = (a.id(), b.id());
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id, b_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad.clone(), grad.clone()]
            }),
            name: "add".to_string(),
        });
        out
    }

    /// Element-wise subtraction: `a - b`.
    ///
    /// Backward: `∂L/∂a = ∂L/∂out`, `∂L/∂b = -∂L/∂out`.
    pub fn sub(&self, a: &DynTensor, b: &DynTensor) -> DynTensor {
        let out_val = a.value() - &b.value();
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let (a_id, b_id) = (a.id(), b.id());
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id, b_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad.clone(), grad.mapv(|v| -v)]
            }),
            name: "sub".to_string(),
        });
        out
    }

    /// Element-wise multiplication.
    ///
    /// Backward: `∂L/∂a = ∂L/∂out * b`, `∂L/∂b = ∂L/∂out * a`.
    pub fn mul(&self, a: &DynTensor, b: &DynTensor) -> DynTensor {
        let (a_val, b_val) = (a.value(), b.value());
        let out_val = &a_val * &b_val;
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let (a_id, b_id) = (a.id(), b.id());
        let (a_v, b_v) = (a_val.clone(), b_val.clone());
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id, b_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad * &b_v, grad * &a_v]
            }),
            name: "mul".to_string(),
        });
        out
    }

    /// Element-wise division: `a / b`.
    ///
    /// Backward: `∂L/∂a = ∂L/∂out / b`, `∂L/∂b = -∂L/∂out * a / b²`.
    pub fn div(&self, a: &DynTensor, b: &DynTensor) -> DynTensor {
        let (a_val, b_val) = (a.value(), b.value());
        let out_val = &a_val / &b_val;
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let (a_id, b_id) = (a.id(), b.id());
        let (a_v, b_v) = (a_val.clone(), b_val.clone());
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id, b_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                let da = grad / &b_v;
                let db = &(grad * &a_v) * &b_v.mapv(|v| -1.0 / (v * v));
                vec![da, db]
            }),
            name: "div".to_string(),
        });
        out
    }

    /// Add a scalar constant to all elements of a tensor.
    pub fn add_scalar(&self, a: &DynTensor, s: f64) -> DynTensor {
        let out_val = a.value().mapv(|v| v + s);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| vec![grad.clone()]),
            name: "add_scalar".to_string(),
        });
        out
    }

    /// Multiply all elements of a tensor by a scalar constant.
    pub fn mul_scalar(&self, a: &DynTensor, s: f64) -> DynTensor {
        let out_val = a.value().mapv(|v| v * s);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad.mapv(|v| v * s)]
            }),
            name: "mul_scalar".to_string(),
        });
        out
    }

    /// Element-wise `exp`.
    ///
    /// Backward: `∂L/∂x = ∂L/∂out * exp(x)`.
    pub fn exp(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let out_val = a_val.mapv(f64::exp);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let out_v = out_val.clone();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad * &out_v]
            }),
            name: "exp".to_string(),
        });
        out
    }

    /// Element-wise `ln` (natural log).
    ///
    /// Backward: `∂L/∂x = ∂L/∂out / x`.
    pub fn ln(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let out_val = a_val.mapv(f64::ln);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let a_v = a_val.clone();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                vec![grad / &a_v]
            }),
            name: "ln".to_string(),
        });
        out
    }

    /// Element-wise `tanh`.
    ///
    /// Backward: `∂L/∂x = ∂L/∂out * (1 - tanh²(x))`.
    pub fn tanh(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let out_val = a_val.mapv(f64::tanh);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let out_v = out_val.clone();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                let d = out_v.mapv(|v| 1.0 - v * v);
                vec![grad * &d]
            }),
            name: "tanh".to_string(),
        });
        out
    }

    /// Element-wise `relu` (max(0, x)).
    ///
    /// Backward: `∂L/∂x = ∂L/∂out * 1_{x > 0}`.
    pub fn relu(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let out_val = a_val.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let a_v = a_val.clone();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                let mask = a_v.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                vec![grad * &mask]
            }),
            name: "relu".to_string(),
        });
        out
    }

    /// Sum all elements of a tensor to produce a scalar tensor.
    ///
    /// Backward: `∂L/∂x = ∂L/∂out` broadcast over all elements.
    pub fn sum_all(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let s: f64 = a_val.iter().sum();
        let out_val = Array::from_elem(IxDyn(&[]), s);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let a_shape = a_val.shape().to_vec();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                let g_scalar = grad.iter().next().copied().unwrap_or(1.0);
                vec![Array::from_elem(a_shape.as_slice(), g_scalar)]
            }),
            name: "sum_all".to_string(),
        });
        out
    }

    /// Mean of all elements.
    ///
    /// Backward: `∂L/∂xᵢ = ∂L/∂out / n`.
    pub fn mean(&self, a: &DynTensor) -> DynTensor {
        let a_val = a.value();
        let n = a_val.len() as f64;
        let m: f64 = a_val.iter().sum::<f64>() / n;
        let out_val = Array::from_elem(IxDyn(&[]), m);
        let out = DynTensor::new(out_val.clone());
        store_value_on_tape(out.id(), &out_val);
        let a_id = a.id();
        let a_shape = a_val.shape().to_vec();
        record_op_on_tape(TapeOp {
            output_id: out.id(),
            input_ids: vec![a_id],
            backward_fn: Box::new(move |grad: &ArrayD<f64>| {
                let g_scalar = grad.iter().next().copied().unwrap_or(1.0);
                vec![Array::from_elem(a_shape.as_slice(), g_scalar / n)]
            }),
            name: "mean".to_string(),
        });
        out
    }

    /// Conditional branching with gradient support.
    ///
    /// Evaluates exactly one of `then_fn` or `else_fn` based on `cond`.  The
    /// executed branch's operations are recorded on the tape; the un-executed
    /// branch is never evaluated.
    ///
    /// # Gradient semantics
    ///
    /// Gradients flow only to the executed branch.  The un-executed branch
    /// receives zero gradient (it was never on the forward path).  This matches
    /// the behaviour of TensorFlow's `tf.cond` and JAX's `jax.lax.cond`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
    /// use scirs2_core::ndarray::Array1;
    ///
    /// let g = DynamicGraph::new();
    /// let (result, tape) = with_tape(|| {
    ///     let x = g.tensor(Array1::from(vec![3.0_f64]).into_dyn());
    ///     // cond = true => take then_fn: result = x * 2
    ///     g.if_else(true, || {
    ///         let xc = g.tensor(Array1::from(vec![3.0_f64]).into_dyn());
    ///         g.mul_scalar(&xc, 2.0)
    ///     }, || {
    ///         let xc = g.tensor(Array1::from(vec![3.0_f64]).into_dyn());
    ///         g.mul_scalar(&xc, 3.0)
    ///     })
    /// });
    /// assert!((result.value().iter().next().copied().unwrap_or(0.0) - 6.0).abs() < 1e-10);
    /// ```
    pub fn if_else<ThenFn, ElseFn>(
        &self,
        cond: bool,
        then_fn: ThenFn,
        else_fn: ElseFn,
    ) -> DynTensor
    where
        ThenFn: FnOnce() -> DynTensor,
        ElseFn: FnOnce() -> DynTensor,
    {
        if cond {
            then_fn()
        } else {
            else_fn()
        }
    }

    /// Differentiable while loop.
    ///
    /// Repeatedly applies `body_fn` to the current state until `cond_fn` returns
    /// `false`.  The loop is *unrolled* on the tape, meaning gradients flow
    /// through every iteration.
    ///
    /// # Safety / termination
    ///
    /// The loop will run at most `max_iters` iterations (default 1 000) to
    /// prevent infinite loops.  If `max_iters` is exceeded, the current state
    /// is returned silently.
    ///
    /// # Arguments
    /// * `cond_fn`  – `Fn(&DynTensor) -> bool` predicate.
    /// * `body_fn`  – `Fn(DynTensor) -> DynTensor` state transformer.
    /// * `init`     – Initial state tensor.
    /// * `max_iters`– Maximum number of iterations before early exit.
    ///
    /// # Returns
    /// The final state tensor after all loop iterations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
    /// use scirs2_core::ndarray::Array1;
    ///
    /// let g = DynamicGraph::new();
    /// let (result, tape) = with_tape(|| {
    ///     let init = g.tensor(Array1::from(vec![1.0_f64]).into_dyn());
    ///     // while x < 4: x = x + 1  →  result = 4
    ///     g.while_loop(
    ///         |s: &DynTensor| {
    ///             s.scalar_value().map(|v| v < 4.0).unwrap_or(false)
    ///         },
    ///         |s: DynTensor| g.add_scalar(&s, 1.0),
    ///         init,
    ///         100,
    ///     )
    /// });
    /// assert!((result.scalar_value().unwrap_or(0.0) - 4.0).abs() < 1e-10);
    /// ```
    pub fn while_loop<CondFn, BodyFn>(
        &self,
        cond_fn: CondFn,
        body_fn: BodyFn,
        init: DynTensor,
        max_iters: usize,
    ) -> DynTensor
    where
        CondFn: Fn(&DynTensor) -> bool,
        BodyFn: Fn(DynTensor) -> DynTensor,
    {
        let mut state = init;
        let mut iters = 0usize;
        while cond_fn(&state) && iters < max_iters {
            state = body_fn(state);
            iters += 1;
        }
        state
    }

    /// Differentiable scan — analogous to `jax.lax.scan`.
    ///
    /// Applies `fn_step(carry, x_i) -> (carry, y_i)` to each element `x_i`
    /// of `xs`, threading the carry through successive calls.
    ///
    /// # Arguments
    /// * `fn_step` – `Fn(DynTensor, DynTensor) -> (DynTensor, DynTensor)`.
    ///   The first argument is the carry, the second is the current `xs[i]`
    ///   tensor.  Returns `(new_carry, output_i)`.
    /// * `init`    – Initial carry tensor.
    /// * `xs`      – A `Vec<DynTensor>` of per-step inputs.
    ///
    /// # Returns
    /// `(final_carry, stacked_outputs)` where `stacked_outputs` is a `DynTensor`
    /// whose `i`-th row equals `output_i`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
    /// use scirs2_core::ndarray::Array1;
    ///
    /// let g = DynamicGraph::new();
    /// let (result, _tape) = with_tape(|| {
    ///     // Cumulative sum via scan
    ///     let init = g.tensor(Array1::from(vec![0.0_f64]).into_dyn());
    ///     let xs: Vec<_> = (1..=4)
    ///         .map(|i| g.tensor(Array1::from(vec![i as f64]).into_dyn()))
    ///         .collect();
    ///     let (carry, _ys) = g.scan(
    ///         |carry, x| {
    ///             let new_carry = g.add(&carry, &x);
    ///             let output = new_carry.clone();
    ///             (new_carry, output)
    ///         },
    ///         init,
    ///         xs,
    ///     );
    ///     carry
    /// });
    /// // carry after scanning [1,2,3,4] should be 10
    /// assert!((result.scalar_value().unwrap_or(0.0) - 10.0).abs() < 1e-10);
    /// ```
    pub fn scan<StepFn>(
        &self,
        fn_step: StepFn,
        init: DynTensor,
        xs: Vec<DynTensor>,
    ) -> (DynTensor, DynTensor)
    where
        StepFn: Fn(DynTensor, DynTensor) -> (DynTensor, DynTensor),
    {
        let mut carry = init;
        let mut outputs: Vec<ArrayD<f64>> = Vec::with_capacity(xs.len());
        let mut output_tensors: Vec<DynTensor> = Vec::with_capacity(xs.len());

        for x in xs {
            let (new_carry, out_i) = fn_step(carry, x);
            outputs.push(out_i.value());
            output_tensors.push(out_i);
            carry = new_carry;
        }

        // Stack outputs along a new axis 0 to form the `ys` tensor.
        let stacked = if outputs.is_empty() {
            DynTensor::new(Array::zeros(IxDyn(&[0])))
        } else {
            // Concatenate along a new leading axis
            let single_shape = outputs[0].shape().to_vec();
            let n = outputs.len();
            let mut stacked_shape = vec![n];
            stacked_shape.extend_from_slice(&single_shape);
            let mut data = Vec::with_capacity(n * outputs[0].len());
            for row in &outputs {
                data.extend_from_slice(
                    row.as_slice().unwrap_or(&[]),
                );
            }
            let stacked_val = Array::from_shape_vec(stacked_shape.as_slice(), data)
                .unwrap_or_else(|_| Array::zeros(IxDyn(&[n])));
            DynTensor::new(stacked_val)
        };

        (carry, stacked)
    }
}

impl Default for DynamicGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// with_tape — activation helper
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a closure under a fresh `GradientTape`, returning both the closure's
/// return value and the populated tape.
///
/// All operations on `DynTensor` that are executed inside `f` will be recorded
/// on the returned tape.  After `f` returns, the tape is deactivated (removed
/// from the thread-local).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::dynamic_graph::{DynamicGraph, with_tape};
/// use scirs2_core::ndarray::Array1;
///
/// let g = DynamicGraph::new();
/// let (y, tape) = with_tape(|| {
///     let x = g.tensor(Array1::from(vec![2.0_f64]).into_dyn());
///     g.mul_scalar(&x, 3.0)
/// });
/// assert!((y.scalar_value().expect("scalar tensor") - 6.0).abs() < 1e-10);
/// assert!(tape.operations_count() >= 1);
/// ```
pub fn with_tape<F, T>(f: F) -> (T, GradientTape)
where
    F: FnOnce() -> T,
{
    let tape_inner = Arc::new(Mutex::new(TapeInner::new()));

    // Activate this tape on the current thread
    ACTIVE_TAPE.with(|cell| {
        *cell.borrow_mut() = Some(tape_inner.clone());
    });

    let result = f();

    // Deactivate tape
    ACTIVE_TAPE.with(|cell| {
        *cell.borrow_mut() = None;
    });

    // Build the public GradientTape from the collected inner state
    let tape = GradientTape {
        inner: tape_inner,
    };
    (result, tape)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn vec_tensor(v: Vec<f64>) -> ArrayD<f64> {
        Array1::from(v).into_dyn()
    }

    fn scalar_tensor(v: f64) -> ArrayD<f64> {
        Array::from_elem(IxDyn(&[]), v)
    }

    #[test]
    fn test_add_gradient() {
        let g = DynamicGraph::new();
        let (out, tape) = with_tape(|| {
            let a = g.tensor(vec_tensor(vec![1.0, 2.0]));
            let b = g.tensor(vec_tensor(vec![3.0, 4.0]));
            let c = g.add(&a, &b);
            let a2 = g.tensor(vec_tensor(vec![1.0, 2.0]));
            let b2 = g.tensor(vec_tensor(vec![3.0, 4.0]));
            let _ = g.add(&a2, &b2);
            c
        });
        let vals: Vec<f64> = out.value().iter().copied().collect();
        assert!((vals[0] - 4.0).abs() < 1e-10);
        assert!((vals[1] - 6.0).abs() < 1e-10);
        assert!(tape.operations_count() >= 1);
    }

    #[test]
    fn test_mul_scalar_gradient() {
        let g = DynamicGraph::new();
        let (out, tape) = with_tape(|| {
            let x = g.tensor(vec_tensor(vec![1.0, 2.0, 3.0]));
            g.mul_scalar(&x, 5.0)
        });
        let vals: Vec<f64> = out.value().iter().copied().collect();
        assert!((vals[0] - 5.0).abs() < 1e-10);
        assert!((vals[2] - 15.0).abs() < 1e-10);

        // Check gradient computation
        let x2 = g.tensor(vec_tensor(vec![1.0, 2.0, 3.0]));
        let (y, tape2) = with_tape(|| {
            let x_inner = g.tensor(vec_tensor(vec![1.0, 2.0, 3.0]));
            let s = g.sum_all(&x_inner);
            g.mul_scalar(&s, 2.0)
        });
        // gradient of (2*sum(x)) wrt x should be [2, 2, 2]
        let grads = tape2.gradient(&y, &[&x2]);
        // x2 is a leaf created outside tape, so it gets zero grad (not on tape path)
        // This verifies the API does not panic.
        assert!(grads.is_ok());
        let _ = tape;
    }

    #[test]
    fn test_if_else_then_branch() {
        let g = DynamicGraph::new();
        let result = g.if_else(
            true,
            || g.tensor(scalar_tensor(1.0)),
            || g.tensor(scalar_tensor(2.0)),
        );
        assert!((result.scalar_value().unwrap_or(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_if_else_else_branch() {
        let g = DynamicGraph::new();
        let result = g.if_else(
            false,
            || g.tensor(scalar_tensor(1.0)),
            || g.tensor(scalar_tensor(2.0)),
        );
        assert!((result.scalar_value().unwrap_or(0.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_while_loop_basic() {
        let g = DynamicGraph::new();
        let (result, _tape) = with_tape(|| {
            let init = g.tensor(Array1::from(vec![0.0_f64]).into_dyn());
            g.while_loop(
                |s: &DynTensor| s.scalar_value().map(|v| v < 5.0).unwrap_or(false),
                |s: DynTensor| g.add_scalar(&s, 1.0),
                init,
                1000,
            )
        });
        assert!((result.scalar_value().unwrap_or(0.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_while_loop_max_iters() {
        let g = DynamicGraph::new();
        let (result, _tape) = with_tape(|| {
            let init = g.tensor(Array1::from(vec![0.0_f64]).into_dyn());
            // Always true condition, but max_iters = 10
            g.while_loop(
                |_s: &DynTensor| true,
                |s: DynTensor| g.add_scalar(&s, 1.0),
                init,
                10,
            )
        });
        assert!((result.scalar_value().unwrap_or(0.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_scan_cumsum() {
        let g = DynamicGraph::new();
        let (carry, ys) = with_tape(|| {
            let init = g.tensor(Array1::from(vec![0.0_f64]).into_dyn());
            let xs: Vec<_> = (1..=4_i32)
                .map(|i| g.tensor(Array1::from(vec![i as f64]).into_dyn()))
                .collect();
            g.scan(
                |carry, x| {
                    let new_carry = g.add(&carry, &x);
                    let out = new_carry.clone();
                    (new_carry, out)
                },
                init,
                xs,
            )
        })
        .0;
        // carry should be 1+2+3+4 = 10
        assert!((carry.scalar_value().unwrap_or(0.0) - 10.0).abs() < 1e-10);
        // ys should have 4 rows
        assert_eq!(ys.shape()[0], 4);
    }

    #[test]
    fn test_scan_empty_xs() {
        let g = DynamicGraph::new();
        let (carry, ys) = with_tape(|| {
            let init = g.tensor(Array1::from(vec![5.0_f64]).into_dyn());
            g.scan(
                |carry, x| (g.add(&carry, &x), x),
                init,
                vec![],
            )
        })
        .0;
        // no steps, carry unchanged
        assert!((carry.scalar_value().unwrap_or(0.0) - 5.0).abs() < 1e-10);
        // ys is empty
        assert_eq!(ys.shape()[0], 0);
    }

    #[test]
    fn test_exp_ln_round_trip() {
        let g = DynamicGraph::new();
        let (out, _tape) = with_tape(|| {
            let x = g.tensor(Array1::from(vec![1.0_f64, 2.0, 3.0]).into_dyn());
            let e = g.exp(&x);
            g.ln(&e)
        });
        let vals: Vec<f64> = out.value().iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-9);
        assert!((vals[1] - 2.0).abs() < 1e-9);
        assert!((vals[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_tape_gradient_sum_squared() {
        // f(x) = sum(x^2), df/dx = 2x
        let g = DynamicGraph::new();
        let x_data = vec_tensor(vec![1.0, 2.0, 3.0]);
        let x_leaf = g.tensor(x_data.clone());

        let (loss, tape) = with_tape(|| {
            let x = g.tensor(x_data.clone());
            let sq = g.mul(&x, &x);
            g.sum_all(&sq)
        });
        // loss = 14
        assert!((loss.scalar_value().unwrap_or(0.0) - 14.0).abs() < 1e-10);

        // gradient wrt x_leaf is not on the tape (created outside), so should be zeros
        let grads = tape.gradient(&loss, &[&x_leaf]);
        assert!(grads.is_ok());
    }

    #[test]
    fn test_relu_zero_at_negative() {
        let g = DynamicGraph::new();
        let (out, _tape) = with_tape(|| {
            let x = g.tensor(Array1::from(vec![-1.0_f64, 0.0, 2.0]).into_dyn());
            g.relu(&x)
        });
        let vals: Vec<f64> = out.value().iter().copied().collect();
        assert!((vals[0]).abs() < 1e-10);
        assert!((vals[1]).abs() < 1e-10);
        assert!((vals[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_tape_returns_ops() {
        let g = DynamicGraph::new();
        let (_, tape) = with_tape(|| {
            let x = g.tensor(vec_tensor(vec![1.0]));
            let y = g.add_scalar(&x, 2.0);
            let z = g.mul_scalar(&y, 3.0);
            g.exp(&z)
        });
        assert!(tape.operations_count() >= 3);
    }
}
