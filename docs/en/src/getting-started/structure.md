# Project Structure

SciRS2 is organized as a Cargo workspace containing roughly 29 crates.
Each crate can be used independently, so you only pull in what you need.

## Directory Layout

```
scirs/
├── Cargo.toml          # Workspace root
├── scirs2-core/        # Foundation crate (ndarray re-export, shared types)
├── scirs2-linalg/      # Linear algebra
├── scirs2-stats/       # Statistics
├── scirs2-signal/      # Signal processing
├── scirs2-fft/         # Fast Fourier transforms
├── scirs2-optimize/    # Optimization
├── scirs2-integrate/   # Numerical integration and ODEs/PDEs
├── scirs2-interpolate/ # Interpolation
├── scirs2-special/     # Special functions
├── scirs2-sparse/      # Sparse matrices
├── scirs2-ndimage/     # Image processing
├── scirs2-neural/      # Neural networks
├── scirs2-graph/       # Graph neural networks
├── scirs2-series/      # Time series analysis
├── scirs2-text/        # NLP and text processing
├── scirs2-vision/      # Computer vision
├── scirs2-io/          # Data I/O (Parquet, Arrow, Zarr, etc.)
├── scirs2-datasets/    # Datasets and data loading
├── scirs2-metrics/     # Evaluation metrics
├── scirs2-cluster/     # Clustering algorithms
├── scirs2-transform/   # Data transformations
├── scirs2-wasm/        # WebAssembly bindings
└── docs/               # Documentation (this book)
```

## Crate Dependency Graph

Every crate depends on `scirs2-core`, which re-exports `ndarray` and `num-complex`
and provides shared error types and utility functions.

```
scirs2-core  ← foundation for all crates
├── scirs2-linalg
│   ├── scirs2-sparse (sparse linear algebra)
│   └── scirs2-integrate (numerical solvers using LA)
├── scirs2-stats
│   └── scirs2-metrics (statistical metrics)
├── scirs2-fft
│   └── scirs2-signal (FFT-based signal processing)
└── scirs2-neural
    └── scirs2-graph (graph neural networks)
```

## Version Management

All crates share a single version number managed in the workspace root `Cargo.toml`:

```toml
# Root Cargo.toml
[workspace.package]
version = "0.4.0"
```

Each crate's `Cargo.toml` references it with `version.workspace = true`.

## Pure Rust Policy

The default SciRS2 build has zero C/Fortran dependencies. Features that require
external runtimes (e.g., GPU) are isolated behind feature flags.

| Pure Rust Library | Replaces |
|-------------------|----------|
| OxiBLAS | OpenBLAS / MKL |
| OxiFFT | FFTW / RustFFT |
| OxiARC | zlib / zstd / flate2 |
| OxiCode | bincode |

This design means `cargo build` is all you need on any platform, including
cross-compilation targets.
