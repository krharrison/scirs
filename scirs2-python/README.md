# SciRS2 - Python Bindings

**SciRS2**: Type-safe scientific computing in Rust with Python bindings - Specialized for complex statistical analysis with exceptional performance for higher-order statistics.

[![PyPI](https://img.shields.io/pypi/v/scirs2)](https://pypi.org/project/scirs2/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/scirs2)](https://pypi.org/project/scirs2/)

## Overview

SciRS2-Python provides Python bindings for the SciRS2 scientific computing ecosystem, offering:

- **Exceptional Complex Statistics**: Up to **410x faster** than SciPy for skewness, kurtosis, and higher-order moments on small datasets
- **Type Safety**: Rust's compile-time guarantees prevent many runtime errors
- **SciPy-Compatible APIs**: Familiar interface for Python scientists
- **Zero-Copy Integration**: Efficient NumPy array interoperability
- **Pure Rust Linear Algebra**: OxiBLAS-powered BLAS/LAPACK operations (no system dependencies)
- **OxiFFT Integration**: Pure Rust high-performance FFT with FFTW-compatible algorithms
- **Hybrid Approach**: Use alongside NumPy/SciPy for optimal performance

**Important**: SciRS2 is a **specialized tool** for type-safe complex statistical analysis, not a general-purpose NumPy/SciPy replacement. See [Performance Guide](#performance) for when to use scirs2 vs NumPy.

## Pure Rust Implementation (v0.1.5+)

**SciRS2 now uses OxiBLAS** - a pure Rust BLAS/LAPACK implementation. No system dependencies required.

| Platform | Backend | Performance Level |
|----------|---------|-------------------|
| **macOS** | OxiBLAS (pure Rust) | ✅ Good |
| **Linux** | OxiBLAS (pure Rust) | ✅ Good |
| **Windows** | OxiBLAS (pure Rust) | ✅ Good |

### Benefits
- ✅ **Zero System Dependencies** - No need to install OpenBLAS, MKL, or Accelerate
- ✅ **Cross-Platform Consistency** - Same behavior on all platforms
- ✅ **Simple Installation** - Just `pip install scirs2`

## Installation

```bash
pip install scirs2
```

For development:
```bash
pip install scirs2[dev]
```

## Quick Start

**SciRS2 excels at statistics** - up to 410x faster than SciPy on complex operations:

```python
import numpy as np
import scirs2

# Generate data
data = np.random.randn(1000)

# ✅ Use scirs2 for ALL statistics on small-medium data (MUCH FASTER!)
mean = scirs2.mean_py(data)          # 8x faster than NumPy!
std = scirs2.std_py(data, 0)         # 14x faster than NumPy!
skewness = scirs2.skew_py(data)      # 52x faster than SciPy!
kurtosis = scirs2.kurtosis_py(data)  # 52x faster than SciPy!

print(f"Mean: {mean:.4f}, Std: {std:.4f}")
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

# ✅ Linear algebra: competitive with BLAS
A = np.random.randn(50, 50)
det = scirs2.det_py(A)               # ~1x vs SciPy (uses Accelerate/OpenBLAS)
inv = scirs2.inv_py(A)               # ~1x vs SciPy

# ✅ FFT: fast on small data
signal = np.random.randn(512)
rfft = scirs2.rfft_py(signal)        # 3x faster than NumPy!
```

## Modules (v0.2.0)

### Linear Algebra (`linalg`)

Linear algebra operations use OxiBLAS (pure Rust). No system BLAS installation required.

```python
import numpy as np
import scirs2

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([1.0, 2.0])

# Basic operations
det = scirs2.det_py(A)           # Determinant
inv = scirs2.inv_py(A)           # Inverse
trace = scirs2.trace_py(A)       # Trace

# Decompositions
lu = scirs2.lu_py(A)             # LU: {'L', 'U', 'P'}
qr = scirs2.qr_py(A)             # QR: {'Q', 'R'}
svd = scirs2.svd_py(A)           # SVD: {'U', 'S', 'Vt'}
chol = scirs2.cholesky_py(A)     # Cholesky

# Eigenvalues
eig = scirs2.eig_py(A)           # {'eigenvalues_real', 'eigenvalues_imag', 'eigenvectors'}
eigh = scirs2.eigh_py(A)         # For symmetric matrices

# Linear systems
x = scirs2.solve_py(A, b)        # Solve Ax = b
lstsq = scirs2.lstsq_py(A, b)    # Least squares

# Norms
norm_fro = scirs2.matrix_norm_py(A, "fro")
norm_vec = scirs2.vector_norm_py(b, 2)
cond = scirs2.cond_py(A)
rank = scirs2.matrix_rank_py(A)
```

### Statistics (`stats`)

✅ **Performance Strength**: Average **30x faster** than SciPy! Complex statistics (skewness, kurtosis) are up to **410x faster** on small datasets. Even basic stats (mean, std) are 5-25x faster on small-medium data (<10K elements).

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Descriptive statistics
stats = scirs2.describe_py(data)  # Returns dict with all stats
mean = scirs2.mean_py(data)
std = scirs2.std_py(data, 0)      # ddof=0 for population
var = scirs2.var_py(data, 1)      # ddof=1 for sample

# Percentiles
median = scirs2.median_py(data)
p75 = scirs2.percentile_py(data, 75.0)
iqr = scirs2.iqr_py(data)

# Correlation
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
corr = scirs2.correlation_py(x, y)  # Returns 1.0
cov = scirs2.covariance_py(x, y, 1)
```

### FFT (`fft`)

FFT operations use OxiFFT backend (Pure Rust, FFTW-compatible algorithms). Performance is **2-5x faster** than NumPy on small data (<2K samples), but NumPy is faster on large data (>32K samples).

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# FFT
result = scirs2.fft_py(data)      # {'real', 'imag'}
real, imag = result['real'], result['imag']

# Inverse FFT
reconstructed = scirs2.ifft_py(
    np.array(real), np.array(imag)
)

# Real FFT (for real-valued signals)
rfft = scirs2.rfft_py(data)
irfft = scirs2.irfft_py(
    np.array(rfft['real']),
    np.array(rfft['imag']),
    len(data)
)

# DCT
dct = scirs2.dct_py(data, 2)      # Type-II DCT
idct = scirs2.idct_py(np.array(dct), 2)

# Helper functions
freqs = scirs2.fftfreq_py(len(data), 1.0)
rfreqs = scirs2.rfftfreq_py(len(data), 1.0)
shifted = scirs2.fftshift_py(data)
fast_len = scirs2.next_fast_len_py(100, False)
```

### Clustering (`cluster`)

```python
import numpy as np
import scirs2

# Generate sample data
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
])

# K-Means clustering
kmeans = scirs2.KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels
inertia = kmeans.inertia_

# Evaluation metrics
silhouette = scirs2.silhouette_score_py(X, labels)
davies_bouldin = scirs2.davies_bouldin_score_py(X, labels)
calinski = scirs2.calinski_harabasz_score_py(X, labels)

# Preprocessing
X_std = scirs2.standardize_py(X, True)   # Zero mean, unit variance
X_norm = scirs2.normalize_py(X, "l2")    # L2 normalization
```

### Time Series (`series`)

```python
import numpy as np
import scirs2

# Create time series
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ts = scirs2.PyTimeSeries(data, None)

# Descriptive statistics
stats = ts.describe()

# Differencing
diff1 = scirs2.apply_differencing(ts, 1)        # First difference
seasonal = scirs2.apply_seasonal_differencing(ts, 4)  # Seasonal

# ARIMA modeling
arima = scirs2.PyARIMA(1, 1, 0)  # AR(1), I(1), MA(0)
arima.fit(ts)
forecast = arima.forecast(5)
params = arima.get_params()
print(arima.summary())

# Box-Cox transformation
result = scirs2.boxcox_transform(ts, None)  # Auto-select lambda
transformed = result['transformed']
lambda_val = result['lambda']
recovered = scirs2.boxcox_inverse(np.array(transformed), lambda_val)

# Stationarity test
adf = scirs2.adf_test(ts, None)
print(f"ADF statistic: {adf['statistic']}, p-value: {adf['p_value']}")

# STL decomposition
decomp = scirs2.stl_decomposition(ts, 4)
trend = decomp['trend']
seasonal = decomp['seasonal']
residual = decomp['residual']
```

## Performance

**Benchmark Summary** (macOS Apple Silicon with Accelerate/OxiFFT):
- **Statistics**: **30.40x average speedup**, 85.3% win rate
- **FFT**: **2.24x average speedup**, 53.3% win rate
- **Linear Algebra**: **1.94x slower** (with proper BLAS), competitive on small matrices
- **Strength**: Complex statistics, small-medium data

### Where SciRS2 Excels 🏆

| Operation | Data Size | Speedup | Use Case |
|-----------|-----------|---------|----------|
| **Skewness** | 100 | **410x** | Distribution shape analysis |
| **Kurtosis** | 100 | **408x** | Distribution tail analysis |
| **Skewness** | 1,000 | **52x** | Higher-order moments |
| **Kurtosis** | 1,000 | **52x** | Higher-order moments |
| **Pearson correlation** | 100 | **127x** | Small dataset correlation |
| **Pearson correlation** | 10,000 | **17x** | Medium dataset correlation |
| **IQR** | 100 | **95x** | Quartile calculations |
| **Percentile** | 100 | **49x** | Distribution analysis |
| **Std** | 100 | **25x** | Small data variability |
| **Mean** | 100 | **11x** | Small data average |
| **FFT (rfft)** | 128 | **5.2x** | Small signal processing |
| **FFT (dct)** | 128 | **5.5x** | Small signal processing |
| **Linear Solve** | 10x10 | **9.4x** | Small linear systems |

**Best Use Cases**:
- Complex statistical analysis on datasets <10,000 elements
- Higher-order moments (skewness, kurtosis) - up to **410x faster**
- Correlation analysis - up to **127x faster**
- Distribution shape analysis
- Small FFT operations (<2048 samples)
- Type-safe Rust integration

### Where NumPy/SciPy May Win ⚠️

| Operation | Size | Performance | Notes |
|-----------|------|-------------|-------|
| **Linear algebra** (SVD, QR) | Large (200x200+) | 2-4x slower | SciPy LAPACK more optimized |
| **FFT** operations | Large (32K+) | 1.5-3x slower | NumPy highly optimized for large FFT |
| **Basic stats** | 100K+ | SIMD: 2-4x slower | NumPy C optimization |

**Use NumPy/SciPy for**:
- Large FFT operations (>32K samples)
- Large matrix decompositions (SVD, QR on 200x200+)
- Basic statistics on very large datasets (>100K elements)

**Linear algebra performance**: With proper system BLAS, SciRS2 is within 2x of SciPy. Small matrices (10x10) can be **9x faster**.

### Decision Matrix

| Your Use Case | Recommended Tool | Reason |
|---------------|------------------|--------|
| **Skewness/Kurtosis** | ✅ **scirs2** | 50-410x faster |
| **Correlation** | ✅ **scirs2** | 17-127x faster |
| **Basic stats, small data** (<10K) | ✅ **scirs2** | 3-25x faster |
| **FFT, small data** (<2K) | ✅ **scirs2** | 2-5x faster |
| **Linear algebra, small** (<50x50) | ✅ **scirs2** | 1-9x faster |
| **Linear algebra, large** (200x200+) | ⚡ Either | ~2x slower with BLAS |
| **FFT, large data** (>32K) | ❌ NumPy | 1.5-3x faster |
| **Basic stats, huge data** (>100K) | ❌ NumPy | SIMD optimized |
| **Type-safe Rust integration** | ✅ **scirs2** | Native Rust types |

### Recommended Hybrid Approach

```python
import numpy as np
import scirs2

# Generate data
data = np.random.randn(1000)

# ✅ Use scirs2 for complex stats (MUCH FASTER!)
skewness = scirs2.skew_py(data)      # 52x faster than SciPy!
kurtosis = scirs2.kurtosis_py(data)  # 52x faster than SciPy!

# ✅ Use scirs2 for basic stats on small-medium data
mean = scirs2.mean_py(data)          # 8x faster on 1K elements
std = scirs2.std_py(data, 0)         # 14x faster on 1K elements

# ✅ Linear algebra: excellent with BLAS
matrix = np.random.randn(50, 50)
det = scirs2.det_py(matrix)          # ~1x (comparable to SciPy)
inv = scirs2.inv_py(matrix)          # ~1x (comparable to SciPy)

# ✅ FFT: fast on small data
signal = np.random.randn(512)
rfft = scirs2.rfft_py(signal)        # 3x faster than NumPy!

# ⚠️ For large FFT, NumPy is faster
large_signal = np.random.randn(65536)
spectrum = np.fft.rfft(large_signal) # NumPy wins on large data
```

**Note**: Benchmarks performed on macOS Apple Silicon with Accelerate framework. Performance may vary on other platforms depending on BLAS installation.

## Type Hints

SciRS2 includes type stubs (`.pyi` files) for better IDE support:

```python
# Your IDE will show type hints and autocompletion
import scirs2
result = scirs2.det_py(matrix)  # IDE knows this returns float
```

## Development

### Building from Source

**Prerequisites** (for optimal linear algebra performance):

**macOS**: No additional dependencies - uses Accelerate framework.

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install libopenblas-dev liblapack-dev gfortran
```

**Linux (RHEL/CentOS/Fedora)**:
```bash
sudo dnf install openblas-devel lapack-devel gcc-gfortran
```

**Build Steps**:
```bash
# Install Rust and maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Build (ensure BLAS is installed first!)
cd scirs2-python
maturin develop --release

# Run tests
pip install pytest numpy
pytest tests/
```

**Note**: If you see linker errors about `openblas` or `lapack`, install the system BLAS libraries as shown above.

### Project Structure

```
scirs2-python/
├── src/           # Rust source with PyO3 bindings
│   ├── lib.rs     # Module registration
│   ├── cluster.rs # Clustering bindings
│   ├── series.rs  # Time series bindings
│   ├── linalg.rs  # Linear algebra bindings
│   ├── stats.rs   # Statistics bindings
│   └── fft.rs     # FFT bindings
├── tests/         # Python tests
├── scirs2.pyi     # Type stubs
└── pyproject.toml
```

## Related Projects

- [SciRS2](https://github.com/cool-japan/scirs) - Core Rust library
- [NumPy](https://numpy.org/) - Array operations
- [SciPy](https://scipy.org/) - Scientific Python (API inspiration)

## License

Licensed under the Apache License 2.0. See LICENSE for details.
