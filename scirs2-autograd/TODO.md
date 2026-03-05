# scirs2-autograd TODO

## Status: v0.3.0 Released (February 26, 2026)

## v0.3.0 Completed

### Core Automatic Differentiation
- Reverse-mode AD (VJP / backpropagation) via tape-based gradient accumulation
- Forward-mode AD (JVP / Jacobian-vector products)
- Dynamic computation graph construction
- Lazy evaluation with graph-level optimizations (constant folding, CSE, loop fusion)
- Higher-order derivatives: Hessian, Hessian-vector products
- Second-order optimization support

### Gradient Utilities
- Finite difference numerical differentiation (forward, central, backward)
- Richardson extrapolation for higher-order accuracy
- Gradient checking / numerical verification
- `numerical_diff.rs` module for standalone finite differences

### Memory Optimization
- Gradient checkpointing (recompute-based; `checkpoint`, `adaptive_checkpoint`)
- Checkpoint groups for multi-output operations (`CheckpointGroup`)
- Checkpoint profiler (`CheckpointProfiler` with memory-saved tracking)
- Memory pooling and in-place operations

### Functional Transforms
- `grad` - scalar gradient computation
- `jacobian` - full Jacobian
- `hessian` - second-order derivatives
- `functional_transforms.rs`: vmap-like batching, compose, grad transform

### Implicit Differentiation
- Implicit function theorem-based gradients (`implicit_diff.rs`)
- Fixed-point iteration gradients
- Support for bi-level optimization

### JVP / VJP
- Explicit `jvp` (Jacobian-vector product, forward-mode)
- Explicit `vjp` (vector-Jacobian product, reverse-mode)
- `jvp_vjp.rs` module with composable interfaces

### Differentiable Operations
- Complete arithmetic with broadcasting (add, sub, mul, div, pow)
- Linear algebra with gradients: matmul, inverse, determinant
- Matrix decompositions with gradients: QR, SVD, Cholesky, LU
- Matrix functions: exp, log, sqrt, power, matrix exponential
- Activation functions: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish
- Loss functions: MSE, cross-entropy, sparse categorical cross-entropy
- Convolution: Conv2D, transposed conv, max/avg pooling
- Tensor manipulation: reshape, slice, concat, pad, advanced indexing
- Reductions: sum, mean, max, min, variance

### Mixed Precision
- FP16 / FP32 mixed precision gradient computation (`mixed_precision.rs`)
- Loss scaling for numeric stability

### Lazy Evaluation
- Deferred execution model (`lazy_eval.rs`)
- JIT-like element-wise operation fusion (`jit_fusion.rs`)

### Optimizers
- SGD (with momentum and Nesterov)
- Adam, AdamW
- AdaGrad, RMSprop
- Plain optimizers API (`plain_optimizers.rs`)
- Learning rate schedulers: step, exponential, cosine annealing
- Gradient clipping (norm-based and value-based)
- Namespace-based variable management

### Higher-Order AD
- `higher_order_new.rs` and `higher_order_advanced.rs`
- Efficient Hessian computation
- Hessian-vector products for Newton-CG and trust-region methods

### Debugging and Visualization
- Computation graph visualization via DOT format (`graph_viz.rs`)
- Gradient tape inspection (`tape/`)
- NaN/Inf detection hooks (`debugging.rs`)

### Custom Gradients
- User-defined gradient rules (`custom_grad.rs`, `custom_grad_advanced.rs`)
- `diff_rules.rs` for registering custom derivative rules

### Distributed Gradients
- Gradient aggregation across workers (`distributed_grad.rs`)
- All-reduce primitives

## v0.4.0 Roadmap

### Source-to-Source Transformation
- Source code transformation for AD (compile-time differentiation)
- Operator overloading with compile-time graph construction

### XLA-Like Compilation
- Computation graph lowering to an IR
- XLA-style device placement and fusion

### Symbolic Differentiation
- CAS-style symbolic derivative rules
- Simplification of symbolic expressions before evaluation

### Improved JIT
- Cross-operation fusion across different op types
- Profile-guided optimization for hot paths

### Sparse Gradients
- Sparse tensor representation in the gradient tape
- Efficient sparse-dense gradient accumulation

## Known Issues / Technical Debt

- Some gradient implementations for exotic matrix functions use approximate gradients; exact gradients tracked in issue backlog
- The `jit_fusion.rs` module currently handles only element-wise ops; extension to reduction ops planned
- `graph_viz.rs` DOT output works best for small graphs; large graph layout needs truncation heuristics
