# scirs2-python TODO

## Status: v0.3.2 Released (March 17, 2026)

## v0.3.2 Completed

### Infrastructure
- [x] PyO3-based Python/Rust interop layer
- [x] Maturin build system (`pyproject.toml`)
- [x] `scirs2-numpy` integration for native ndarray 0.17 support
- [x] Zero-copy NumPy array sharing via `scirs2-numpy` bridge
- [x] Type stubs (`scirs2.pyi`) for IDE autocompletion
- [x] Feature-gated modules for selective SciRS2 crate inclusion
- [x] `async_ops.rs` - Async Python-compatible operations
- [x] `error.rs` - Unified error type translating Rust errors to Python exceptions
- [x] `pandas_compat.rs` - pandas DataFrame interop utilities

### Linear Algebra (`linalg.rs`, `linalg_ext.rs`)
- [x] `det_py`, `inv_py`, `trace_py` - Basic matrix properties
- [x] `lu_py`, `qr_py`, `svd_py`, `cholesky_py` - Decompositions
- [x] `eig_py`, `eigh_py` - Eigenvalues and eigenvectors
- [x] `solve_py`, `lstsq_py` - Linear solvers
- [x] `matrix_norm_py`, `vector_norm_py`, `cond_py`, `matrix_rank_py`
- [x] Extended: `expm_py` (matrix exponential), `logm_py`, `sqrtm_py`
- [x] Extended: `schur_py`, `kron_py`, `block_diag_py`
- [x] OxiBLAS backend — no system OpenBLAS needed

### Statistics (`stats.rs`, `stats/mcmc_gp.rs`)
- [x] Descriptive: `mean_py`, `std_py`, `var_py`, `median_py`, `percentile_py`, `iqr_py`
- [x] Higher-order moments: `skew_py`, `kurtosis_py`
- [x] Correlation: `correlation_py`, `covariance_py`
- [x] Full summary: `describe_py`
- [x] MCMC: `MetropolisHastings`, `HamiltonianMC`, `NUTS` samplers
- [x] Gaussian Process: `GaussianProcessRegressor` with RBF, Matern, periodic kernels
- [x] Survival analysis: `KaplanMeier`, `CoxPH`, `NelsonAalen`

### FFT (`fft.rs`)
- [x] `fft_py`, `ifft_py` - Complex FFT
- [x] `rfft_py`, `irfft_py` - Real FFT
- [x] `dct_py`, `idct_py` - Discrete cosine transform (types I-IV)
- [x] `fftfreq_py`, `rfftfreq_py`, `fftshift_py`, `ifftshift_py`
- [x] `next_fast_len_py`
- [x] OxiFFT backend — no system FFTW needed

### Clustering (`cluster.rs`)
- [x] `KMeans` - K-Means clustering
- [x] `silhouette_score_py`, `davies_bouldin_score_py`, `calinski_harabasz_score_py`
- [x] `standardize_py`, `normalize_py`
- [x] DBSCAN, Hierarchical clustering bindings

### Time Series (`series.rs`)
- [x] `PyTimeSeries` - Time series container
- [x] `PyARIMA` - ARIMA modelling (fit, forecast, summary)
- [x] `apply_differencing`, `apply_seasonal_differencing`
- [x] `boxcox_transform`, `boxcox_inverse`
- [x] `adf_test` - Augmented Dickey-Fuller stationarity test
- [x] `stl_decomposition` - STL trend/seasonal/residual decomposition

### Signal Processing (`signal.rs`, `signal_ext.rs`)
- [x] Filter design: `butter_py`, `cheby1_py`, `cheby2_py`
- [x] Filter application: `lfilter_py`, `sosfilt_py`
- [x] Spectrogram, STFT, periodogram
- [x] Extended: Kalman filter, EKF, UKF Python bindings
- [x] Extended: adaptive filter (LMS, RLS) bindings

### Optimization (`optimize.rs`, `optimize_ext.rs`)
- [x] Unconstrained: Nelder-Mead, BFGS, L-BFGS-B, CG
- [x] Constrained: SLSQP with equality and inequality constraints
- [x] Global: differential evolution, basin-hopping
- [x] Curve fitting: `curve_fit_py`
- [x] Extended: SQP, interior-point LP/QP bindings

### Other Modules
- [x] `spatial.rs` - KD-tree, ball tree, distance metrics
- [x] `sparse.rs` - CSR/CSC sparse matrix ops
- [x] `ndimage.rs` - Gaussian blur, morphology, labeling
- [x] `graph.rs` - Graph construction, shortest paths, community detection
- [x] `metrics.rs` - Classification, regression, clustering metrics
- [x] `io.rs` - CSV, HDF5, Parquet, Arrow read/write
- [x] `datasets.rs` - `load_iris_py`, `load_boston_py`, `make_classification_py`, etc.
- [x] `transform.rs` - PCA, ICA, t-SNE, UMAP bindings
- [x] `text.rs` - Tokenization, TF-IDF, Word2Vec, NER bindings
- [x] `vision.rs` - Image processing, feature detection bindings

### Testing
- [x] `tests/test_module_structure.py` - Module import and structure tests
- [x] Basic numerical correctness tests for each module
- [x] SciPy comparison tests for statistics and FFT

## v0.4.0 Roadmap

### Full API Coverage
- [ ] Complete parity with all scirs2-linalg functions
- [ ] Complete parity with all scirs2-stats distributions
- [ ] `scirs2.special` module: Bessel, Gamma, hypergeometric functions
- [ ] `scirs2.interpolate` module: spline, RBF, PCHIP
- [ ] `scirs2.integrate` module: ODE solvers (RK45, BDF, LSODA), quadrature

### Async Python Support
- [ ] Native `async/await` for long-running computations
- [ ] `asyncio`-compatible interface using `pyo3-asyncio`
- [ ] Parallel batch processing with Python threads

### GPU Tensor Bridge
- [ ] Optional CUDA tensor bridge via `cudarc` or `candle`
- [ ] PyTorch tensor interop (zero-copy via DLPack)
- [ ] GPU-accelerated matrix operations exposed to Python

### Type System Improvements
- [ ] `Protocol`-based type stubs for duck-typed APIs
- [ ] Full `mypy`-compatible stubs for all modules
- [ ] Auto-generated stubs from PyO3 introspection

### Packaging and Distribution
- [ ] Pre-built wheels for `manylinux2014`, `musllinux`, `macOS-arm64`, `macOS-x86_64`, `win-amd64`
- [ ] GitHub Actions release pipeline via Maturin's `zig` cross-compilation
- [ ] PyPI publishing automation

### Documentation
- [ ] Sphinx API documentation with `maturin-sphinx` plugin
- [ ] SciPy migration guide with side-by-side examples
- [ ] Performance comparison notebooks (Jupyter)

## Known Issues

- ndarray version boundary: `scirs2-numpy` resolves the ndarray 0.16/0.17 mismatch that blocked earlier versions; this is fully resolved in v0.3.1.
- Large matrix operations (>200x200) may be slower than SciPy with a well-tuned system LAPACK; use NumPy/SciPy for those cases.
- `scirs2-python` is excluded from the default workspace build (`--exclude scirs2-python`) because it requires Python dev headers.
- Graph module suppresses `#[allow(deprecated)]` for `PyAnyMethods::downcast`; will be updated when PyO3 stabilizes the replacement.
