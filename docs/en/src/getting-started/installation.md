# Installation

## Requirements

- Rust 1.75 or later (2021 edition)
- No C/Fortran compiler needed -- SciRS2 is pure Rust by default

## Adding SciRS2 to Your Project

The simplest way to get started is to depend on the umbrella crate, which re-exports all modules:

```toml
[dependencies]
scirs2 = "0.4.0"
```

For finer-grained control, depend on individual crates:

```toml
[dependencies]
scirs2-core = "0.4.0"
scirs2-linalg = "0.4.0"
scirs2-stats = "0.4.0"
scirs2-fft = "0.4.0"
```

All crates share the same version number via the workspace `version.workspace = true` mechanism.

## Feature Flags

Most crates provide optional feature flags for extended functionality:

| Feature | Description | Available in |
|---------|-------------|--------------|
| `simd` | SIMD-accelerated operations (AVX/AVX2/AVX-512) | linalg, signal, fft |
| `parallel` | Multi-threaded execution via Rayon | linalg, fft, sparse |
| `gpu` | GPU acceleration (CUDA/ROCm/Metal/OpenCL) | linalg, fft, sparse, neural |
| `serde` | Serialization support | core, stats, sparse |

Example with features enabled:

```toml
[dependencies]
scirs2-linalg = { version = "0.4.0", features = ["simd", "parallel"] }
scirs2-fft = { version = "0.4.0", features = ["parallel"] }
```

## The Core Crate

`scirs2-core` is the foundation crate that all other SciRS2 crates depend on. It re-exports
`ndarray` and `num-complex` so you do not need to add them separately:

```rust
use scirs2_core::ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
```

This ensures version consistency across the workspace. Always import `ndarray` types through
`scirs2_core` rather than adding a direct `ndarray` dependency.

## Verifying the Installation

Create a small test program:

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::{det, inv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = array![[4.0, 2.0], [2.0, 3.0]];
    let d = det(&a.view(), None)?;
    println!("Determinant: {}", d);

    let a_inv = inv(&a.view(), None)?;
    println!("Inverse:\n{}", a_inv);

    Ok(())
}
```

Run with `cargo run`. If it prints the determinant (8.0) and inverse matrix, everything is
working correctly.

## Platform Support

SciRS2 compiles on all platforms that Rust supports:

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64, aarch64) | Full support | CI tested |
| macOS (x86_64, Apple Silicon) | Full support | CI tested |
| Windows (x86_64) | Full support | CI tested |
| WebAssembly (wasm32) | Via `scirs2-wasm` | Browser and Node.js |
| iOS (aarch64) | Core modules | Min iOS 13.0 |
| Android (aarch64, armv7) | Core modules | Min API 21 |

GPU features require the corresponding runtime (CUDA toolkit, ROCm, etc.) but are
entirely optional and never required for compilation.
