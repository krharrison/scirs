# scirs2-wasm TODO

## Status: v0.3.3 Released (March 17, 2026)

## v0.3.3 Completed

### Core Interface
- [x] `wasm-bindgen` based interface with full JS/TS interop
- [x] `WasmArray` type for 1D/ND array operations
- [x] `WasmMatrix` type for 2D linear algebra
- [x] TypeScript type definitions (`ts-src/scirs2.ts`)
- [x] WASM module initialization with panic hooks
- [x] Version and capability detection (`has_simd_support()`)
- [x] `PerformanceTimer` for profiling in JS environments

### Linear Algebra (`linalg.rs`)
- [x] Matrix creation: zeros, ones, eye, from_rows, from_shape
- [x] Basic ops: add, subtract, multiply (element-wise), matmul
- [x] Decompositions: LU, QR, SVD, Cholesky, eigenvalue/eigenvector
- [x] Solvers: solve (Ax=b), least squares
- [x] Properties: det, trace, rank, norm (Frobenius, 1, 2, inf), cond
- [x] Matrix inverse and pseudoinverse
- [x] Transpose, reshape, slice

### Signal Processing (`signal.rs`)
- [x] FFT and inverse FFT
- [x] RFFT for real-valued signals
- [x] Spectrogram generation (STFT)
- [x] Periodogram (power spectral density)
- [x] FIR filter design and application (lowpass, highpass, bandpass)
- [x] IIR filter (Butterworth, Chebyshev)
- [x] Convolution and correlation
- [x] Window functions (Hann, Hamming, Blackman, Kaiser)

### Statistics (`stats.rs`)
- [x] Descriptive: mean, std, var, median, min, max, sum, skewness, kurtosis
- [x] Percentiles and quantiles
- [x] Correlation coefficient and covariance
- [x] Histogram computation
- [x] Cumulative sum and product
- [x] Statistical tests: t-test, Shapiro-Wilk normality
- [x] Linear regression with R2, slope, intercept, residuals
- [x] Advanced stats: time series primitives, regression diagnostics

### ML Utilities (`ml.rs`, `stats_advanced.rs`, `stats_descriptive.rs`)
- [x] Activation functions: ReLU, sigmoid, softmax, tanh, GELU, ELU, SiLU
- [x] Loss functions: MSE, cross-entropy, binary cross-entropy, Huber
- [x] Normalization: layer norm, batch norm, instance norm
- [x] Distance metrics: L1, L2, cosine, Manhattan
- [x] K-means clustering (forward pass only)
- [x] PCA projection

### Streaming Processing (`streaming.rs`)
- [x] `StreamingProcessor` for incremental dataset processing
- [x] Online mean, variance estimation (Welford algorithm)
- [x] Windowed statistics (rolling mean, rolling std)
- [x] Reservoir sampling for streaming datasets

### WebWorker Support (`worker.rs`)
- [x] `WorkerMessage` type for postMessage serialization
- [x] Offloadable computation types (FFT, matmul, statistics)
- [x] Async result delivery via Promises

### Advanced Modules
- [x] SIMD-accelerated operations (`simd_ops.rs`) for supported runtimes
- [x] Advanced linear algebra (`linalg_advanced.rs`): Krylov solvers, randomized SVD
- [x] Enhanced signal processing (`signal_enhanced.rs`): adaptive filters
- [x] Advanced statistics (`stats_advanced.rs`): bootstrap, Monte Carlo

## v0.4.0 Roadmap

### WebGPU Compute Shaders
- [ ] WebGPU backend for GPU-accelerated matrix multiply
- [ ] WGSL compute shaders for batch operations
- [ ] Fallback to WASM when WebGPU unavailable
- [ ] Benchmark: target 10x speedup over WASM for large matrices

### SharedArrayBuffer / Zero-Copy
- [ ] Zero-copy array sharing between main thread and workers via `SharedArrayBuffer`
- [ ] `Atomics`-based synchronization for concurrent reads
- [ ] Requires COOP/COEP headers (document in setup guide)

### Streaming Large Datasets
- [ ] Async streaming API for datasets that do not fit in WASM memory
- [ ] Lazy FFT for streaming audio/sensor data
- [ ] Incremental PCA on streaming data

### Expanded Signal Processing
- [ ] Wavelet transform (DWT, CWT) in WASM
- [ ] Short-Time Fourier Transform (STFT) with overlap-add reconstruction
- [ ] Mel-frequency cepstral coefficients (MFCC)

### Usability Improvements
- [ ] Automatic memory management with `FinalizationRegistry`
- [ ] Detailed error messages surfaced to JS with error codes
- [ ] `scirs2-wasm-react` helper hooks package
- [ ] npm publish automation via GitHub Actions

### Testing & CI
- [ ] Playwright-based end-to-end tests in real browsers
- [ ] Benchmarks vs `ml5.js`, `tfjs-wasm` for comparable operations
- [ ] Automated WASM binary size regression tracking

## Known Issues

- `SharedArrayBuffer` requires Cross-Origin Isolation (`COOP`/`COEP` headers); document this requirement in the setup guide.
- Safari SIMD support is partial; `has_simd_support()` returns `false` on affected versions.
- WASM memory cannot be released back to the OS once grown; encourage array reuse for long-running applications.
- WebWorker message passing copies data; zero-copy requires `SharedArrayBuffer` (v0.4.0 target).
