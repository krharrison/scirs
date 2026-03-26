# Migrating from SciPy / NumPy

This guide maps common SciPy and NumPy patterns to their SciRS2 equivalents.
The API surface is intentionally similar, so most translations are mechanical.

## Key Differences

1. **Error handling**: SciPy raises exceptions; SciRS2 returns `Result<T, E>`. Append `?`
   to propagate errors or handle them explicitly.

2. **Array creation**: NumPy uses `np.array()`; SciRS2 uses `ndarray::array![]` or
   `Array1::from_vec()`.

3. **Imports**: SciRS2 array types come through `scirs2_core::ndarray` to ensure version
   consistency.

4. **Views**: SciRS2 functions often take `&ArrayView` rather than owned arrays. Use
   `.view()` to create a view.

5. **No implicit broadcasting for all cases**: While ndarray supports broadcasting, some
   operations require explicit shape matching.

## Array Creation

| NumPy | SciRS2 |
|-------|--------|
| `np.array([1, 2, 3])` | `array![1.0, 2.0, 3.0]` |
| `np.zeros((3, 3))` | `Array2::<f64>::zeros((3, 3))` |
| `np.ones((3, 3))` | `Array2::<f64>::ones((3, 3))` |
| `np.eye(3)` | `Array2::eye(3)` |
| `np.linspace(0, 1, 100)` | `Array1::linspace(0.0, 1.0, 100)` |
| `np.arange(0, 10, 0.1)` | `Array1::range(0.0, 10.0, 0.1)` |
| `np.random.randn(100)` | `distributions::norm(0.0, 1.0)?.rvs(100)?` |

## Linear Algebra

| SciPy | SciRS2 |
|-------|--------|
| `np.linalg.det(A)` | `det(&a.view(), None)?` |
| `np.linalg.inv(A)` | `inv(&a.view(), None)?` |
| `np.linalg.solve(A, b)` | `solve(&a.view(), &b.view(), None)?` |
| `scipy.linalg.lu(A)` | `lu(&a.view(), None)?` |
| `scipy.linalg.qr(A)` | `qr(&a.view(), None)?` |
| `np.linalg.svd(A)` | `svd(&a.view(), true, None)?` |
| `np.linalg.eig(A)` | `eig(&a.view(), None)?` |
| `scipy.linalg.cholesky(A)` | `cholesky(&a.view(), None)?` |
| `scipy.linalg.expm(A)` | `expm(&a.view())?` |
| `np.linalg.norm(x)` | `norm(&x.view(), order)?` |

### Example: Python to Rust

Python:
```python
import numpy as np
from scipy.linalg import solve, lu

A = np.array([[2, 1], [5, 3]])
b = np.array([1, 2])
x = solve(A, b)
P, L, U = lu(A)
```

Rust:
```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::{solve, lu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = array![[2.0, 1.0], [5.0, 3.0]];
    let b = array![1.0, 2.0];
    let x = solve(&a.view(), &b.view(), None)?;
    let (p, l, u) = lu(&a.view(), None)?;
    Ok(())
}
```

## Statistics

| SciPy | SciRS2 |
|-------|--------|
| `np.mean(x)` | `mean(&x.view())?` |
| `np.median(x)` | `median(&x.view())?` |
| `np.std(x, ddof=1)` | `std(&x.view(), 1, None)?` |
| `scipy.stats.norm(0, 1)` | `distributions::norm(0.0, 1.0)?` |
| `dist.pdf(x)` | `dist.pdf(x)` |
| `dist.cdf(x)` | `dist.cdf(x)` |
| `dist.ppf(q)` | `dist.ppf(q)?` |
| `dist.rvs(size=n)` | `dist.rvs(n)?` |
| `scipy.stats.ttest_1samp(x, 5)` | `ttest_1samp(&x.view(), 5.0, Alternative::TwoSided, "propagate")?` |
| `scipy.stats.pearsonr(x, y)` | `pearsonr(&x.view(), &y.view())?` |

## FFT

| SciPy | SciRS2 |
|-------|--------|
| `scipy.fft.fft(x)` | `fft(&x, None)?` |
| `scipy.fft.ifft(X)` | `ifft(&x, None)?` |
| `scipy.fft.rfft(x)` | `rfft(&x, None)?` |
| `scipy.fft.fft2(x)` | `fft2(&x, None, None, None)?` |
| `scipy.fft.fftfreq(n, d)` | `fftfreq(n, d)` |

## Signal Processing

| SciPy | SciRS2 |
|-------|--------|
| `scipy.signal.convolve(a, b, "same")` | `convolve(&a, &b, "same")?` |
| `scipy.signal.butter(4, 0.1)` | `butter(4, &[0.1], FilterType::Lowpass, false, None)?` |
| `scipy.signal.sosfilt(sos, x)` | `sosfilt(&sos, &x)?` |
| `scipy.signal.periodogram(x, fs)` | `periodogram(&x, fs, None, None)?` |
| `scipy.signal.welch(x, fs)` | `welch(&x, fs, None, None, None, None)?` |

## Optimization

| SciPy | SciRS2 |
|-------|--------|
| `minimize(f, x0, method="BFGS")` | `minimize(f, &x0, Method::BFGS, None)?` |
| `minimize(f, x0, method="Nelder-Mead")` | `minimize(f, &x0, Method::NelderMead, None)?` |
| `differential_evolution(f, bounds)` | `differential_evolution(f, &bounds, None)?` |
| `least_squares(f, x0)` | `least_squares(f, &x0, Method::LevenbergMarquardt, None)?` |
| `brentq(f, a, b)` | `brentq(f, a, b, None)?` |

## Integration

| SciPy | SciRS2 |
|-------|--------|
| `quad(f, 0, 1)` | `quad(f, 0.0, 1.0, None)?` |
| `solve_ivp(f, t_span, y0)` | `solve_ivp(f, t_span, &y0, &options)?` |
| `romberg(f, 0, pi)` | `romberg(f, 0.0, PI, None)?` |

## Sparse Matrices

| SciPy | SciRS2 |
|-------|--------|
| `csr_array((data, indices, indptr))` | `CsrArray::from_raw(indptr, indices, data, shape)?` |
| `csr_array((data, (row, col)))` | `CsrArray::from_triplets(&rows, &cols, &data, shape, false)?` |
| `A.dot(x)` | `a.dot(&x)?` |
| `spsolve(A, b)` | `spsolve(&a, &b)?` |
| `eigs(A, k)` | `eigs(&a, k, "LM")?` |

## Common Patterns

### Error Handling

Python exceptions become Rust `Result` types:

```rust
// Python: result = solve(A, b)  # raises LinAlgError
// Rust:
match solve(&a.view(), &b.view(), None) {
    Ok(x) => println!("Solution: {}", x),
    Err(e) => eprintln!("Failed: {}", e),
}

// Or use ? to propagate
let x = solve(&a.view(), &b.view(), None)?;
```

### Type Annotations

SciRS2 requires explicit float types. When in doubt, use `f64`:

```rust
use scirs2_core::ndarray::Array2;

// Explicit type annotation
let a: Array2<f64> = Array2::zeros((3, 3));

// Or annotate the literal
let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
```

### Type Mapping

| Python/NumPy | Rust/SciRS2 |
|-------------|-------------|
| `float` / `np.float64` | `f64` |
| `complex` / `np.complex128` | `Complex64` |
| `np.ndarray` (1D) | `Array1<f64>` |
| `np.ndarray` (2D) | `Array2<f64>` |
| `None` | `Option<T>` |
| Exceptions | `Result<T, E>` |
