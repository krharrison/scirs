# Architecture

## Workspace Organization

SciRS2 is a Cargo workspace with approximately 29 crates. The workspace root `Cargo.toml`
defines shared dependencies, versions, and lint configuration. Individual crates use
`version.workspace = true` to inherit the workspace version.

### Layering

```text
                    scirs2 (umbrella)
                         |
    +--------------------+--------------------+
    |         |          |         |           |
  neural    graph      series    text       vision
    |         |          |         |           |
    +----+----+----+-----+----+---+           |
         |         |          |               |
       linalg    stats      fft             wasm
         |         |          |
         +----+----+----+----+
              |         |
            sparse   special
              |         |
              +----+----+
                   |
                 core
```

`scirs2-core` sits at the bottom and provides:

- Re-exported `ndarray` types for version consistency
- Re-exported `num-complex` for complex arithmetic
- Common error types (`CoreError`, `CoreResult`)
- Shared numeric utilities (constants, precision helpers)
- Concurrent data structures (skip list, B-tree, signal/slot)

All other crates depend on `scirs2-core` and import array types through it.

## Module Internal Structure

Each crate follows a consistent layout:

```text
scirs2-{name}/
  Cargo.toml          # version.workspace = true
  src/
    lib.rs            # Public API, module declarations, doc comments
    error.rs          # Crate-specific error types
    {feature_a}/
      mod.rs          # Sub-module public API
      impl.rs         # Implementation details
    {feature_b}/
      mod.rs
  tests/
    integration.rs    # Integration tests
  benches/
    benchmark.rs      # Criterion benchmarks
```

### Error Handling Pattern

Each crate defines its own error enum and result alias:

```rust,ignore
// scirs2-linalg/src/error.rs
pub enum LinalgError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    SingularMatrix,
    NotPositiveDefinite,
    ConvergenceFailed { iterations: usize },
    InvalidInput(String),
}

pub type LinalgResult<T> = Result<T, LinalgError>;
```

Functions return `LinalgResult<T>` rather than panicking. The `?` operator propagates
errors naturally through call chains.

## Feature Flags

Feature flags control optional functionality:

- **`simd`**: SIMD-accelerated code paths (AVX/AVX2/AVX-512 on x86, NEON on ARM)
- **`parallel`**: Multi-threaded execution via Rayon
- **`gpu`**: GPU acceleration (CUDA, ROCm, Metal, OpenCL) -- always behind a feature gate
  to maintain the pure-Rust default
- **`serde`**: Serialization/deserialization support

GPU features are the only ones that may pull in C dependencies (CUDA toolkit, etc.).
Default features are always 100% pure Rust.

## Coding Patterns

### Builder Pattern for Options

Complex functions use a builder pattern for optional parameters:

```rust,ignore
let options = ODEOptions::default()
    .with_method(ODEMethod::RK45)
    .with_rtol(1e-8)
    .with_atol(1e-10)
    .with_max_steps(10000);
```

### View-Based APIs

Functions accept `&ArrayView` rather than owned arrays to avoid unnecessary cloning:

```rust,ignore
pub fn solve(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    options: Option<&SolveOptions>,
) -> LinalgResult<Array1<f64>> { ... }
```

Callers pass `.view()`:

```rust,ignore
let x = solve(&a.view(), &b.view(), None)?;
```

### Trait-Based Polymorphism

Generic algorithms use traits to support multiple numeric types:

```rust,ignore
pub trait Distribution {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn ppf(&self, q: f64) -> StatsResult<f64>;
    fn mean(&self) -> f64;
    fn var(&self) -> f64;
    fn rvs(&self, size: usize) -> StatsResult<Array1<f64>>;
}
```

## Testing Strategy

- **Unit tests**: Inside `src/` files, testing individual functions
- **Integration tests**: In `tests/` directory, testing cross-module interactions
- **Property-based tests**: Using `proptest` for invariant verification (e.g., `A = LU`)
- **Statistical validation**: In `scirs2-validation`, testing numerical accuracy against
  reference implementations
- **Benchmarks**: Using Criterion, with regression detection via `scripts/bench-regression.sh`

Current test count: 30,000+ tests across the workspace.
