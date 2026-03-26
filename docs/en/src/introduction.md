# SciRS2 -- Scientific Computing in Rust

SciRS2 is a comprehensive scientific computing library for Rust, modeled after Python's SciPy ecosystem.
It provides production-ready implementations of numerical algorithms across linear algebra, statistics,
signal processing, optimization, differential equations, and more -- all in pure Rust with no C or
Fortran dependencies.

## Design Philosophy

SciRS2 is built on several core principles:

- **Pure Rust**: The entire library compiles without C/Fortran toolchains. BLAS operations use
  OxiBLAS, FFTs use OxiFFT, and compression uses OxiARC. This makes cross-compilation
  straightforward and eliminates system dependency headaches.

- **SciPy API familiarity**: Function names, parameter ordering, and return types mirror SciPy
  wherever possible. If you know `scipy.linalg.solve`, you already know `scirs2_linalg::solve`.

- **Type safety without ceremony**: Errors are returned via `Result` types rather than panics.
  The `#[clippy::unwrap_used = "warn"]` lint is enforced workspace-wide, so production code
  handles all failure paths explicitly.

- **Performance by default**: SIMD vectorization, Rayon-based parallelism, and cache-friendly
  data layouts are used throughout. Optional GPU acceleration (CUDA, ROCm, Metal) is available
  behind feature flags.

## Feature Overview

SciRS2 spans roughly 29 workspace crates covering the following domains:

| Domain | Crate | SciPy / Python Equivalent |
|--------|-------|---------------------------|
| Linear algebra | `scirs2-linalg` | `scipy.linalg`, `numpy.linalg` |
| Statistics | `scirs2-stats` | `scipy.stats` |
| Signal processing | `scirs2-signal` | `scipy.signal` |
| Fourier transforms | `scirs2-fft` | `scipy.fft` |
| Optimization | `scirs2-optimize` | `scipy.optimize` |
| Integration / ODEs / PDEs | `scirs2-integrate` | `scipy.integrate` |
| Interpolation | `scirs2-interpolate` | `scipy.interpolate` |
| Special functions | `scirs2-special` | `scipy.special` |
| Sparse matrices | `scirs2-sparse` | `scipy.sparse` |
| Image processing | `scirs2-ndimage` | `scipy.ndimage` |
| Neural networks | `scirs2-neural` | PyTorch / TensorFlow |
| Graph neural networks | `scirs2-graph` | PyG, DGL |
| Time series | `scirs2-series` | statsmodels, Darts |
| NLP / text | `scirs2-text` | HuggingFace, gensim |
| Computer vision | `scirs2-vision` | torchvision, OpenCV |
| Data I/O | `scirs2-io` | pandas, pyarrow |
| Datasets | `scirs2-datasets` | sklearn.datasets, HF datasets |
| Metrics | `scirs2-metrics` | sklearn.metrics, torchmetrics |
| Clustering | `scirs2-cluster` | sklearn.cluster |
| Transforms | `scirs2-transform` | sklearn.preprocessing |
| WebAssembly | `scirs2-wasm` | -- |

## Comparison with SciPy

SciRS2 covers roughly the same algorithmic surface as SciPy, with several additions:

- **Neural network layers and training**: Attention mechanisms, quantization (GPTQ, AWQ, SmoothQuant),
  NAS (DARTS, GDAS, SNAS), and distributed training (pipeline/tensor parallelism).
- **Graph neural networks**: GCN, GAT, GraphSAGE, R-GCN, HGT, GraphGPS, Graphormer, and signed/directed
  graph embeddings.
- **Advanced PDE solvers**: Discontinuous Galerkin, virtual element method, peridynamics, and
  physics-informed neural networks (PINNs).
- **WebAssembly support**: Run FFT, linear algebra, and signal processing in the browser via
  `scirs2-wasm` with WebGPU acceleration.

Where SciPy has the advantage is ecosystem maturity and breadth of third-party integrations.
SciRS2 compensates with compile-time safety, zero-cost abstractions, and the ability to deploy
the same code on servers, embedded devices, and browsers without modification.

## COOLJAPAN Ecosystem

SciRS2 is part of the COOLJAPAN open-source ecosystem, which provides pure-Rust replacements for
common C/Fortran scientific libraries:

| Library | Replaces |
|---------|----------|
| OxiBLAS | OpenBLAS, MKL |
| OxiFFT | FFTW |
| OxiARC | zlib, zstd, bzip2 |
| OxiCode | bincode |
| OxiZ | Z3 SMT solver |

All COOLJAPAN libraries share the same commitment to pure Rust compilation, no unsafe where
avoidable, and broad platform support (Linux, macOS, Windows, WASM, iOS, Android).

## Getting Help

- [GitHub Issues](https://github.com/cool-japan/scirs/issues) -- Bug reports and feature requests
- [API Documentation](https://docs.rs/scirs2) -- Auto-generated rustdoc
- This book -- Tutorials, guides, and migration references
