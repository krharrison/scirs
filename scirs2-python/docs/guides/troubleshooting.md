# Troubleshooting Guide

**Version**: SciRS2-Python 0.2.0
**Last Updated**: 2025-12-29

---

## Quick Diagnostics

### Check Installation

```python
import scirs2
print(f"scirs2 version: {scirs2.__version__}")
print(f"scirs2 location: {scirs2.__file__}")
```

Expected output:
```
scirs2 version: 0.3.1
scirs2 location: /path/to/site-packages/scirs2...
```

### Verify Dependencies

```python
import numpy as np
import scipy
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
```

Minimum requirements:
- NumPy ≥ 1.20
- SciPy ≥ 1.7 (optional, for comparison)

---

## Common Issues

### Issue 1: Import Errors

#### Problem: `ModuleNotFoundError: No module named 'scirs2'`

**Cause**: Package not installed

**Solution**:
```bash
pip install scirs2
```

If building from source:
```bash
cd scirs2-python
maturin develop --release
```

**Verify**:
```python
import scirs2  # Should not raise error
```

---

#### Problem: `ImportError: cannot import name 'function_py'`

**Cause**: Function not available in current version

**Solution**: Check available functions:
```python
import scirs2
print(dir(scirs2))  # List all available functions
```

Check API documentation for correct function name.

---

### Issue 2: Performance Issues

#### Problem: scirs2 is slower than expected

**Possible Causes**:
1. Using wrong operations (linear algebra, FFT)
2. Large datasets
3. Non-contiguous arrays

**Diagnostic**:
```python
import numpy as np
import scirs2
import time

data = np.random.randn(1000)

# Check data size
print(f"Data size: {len(data)}")

# Check if contiguous
print(f"Contiguous: {data.flags['C_CONTIGUOUS']}")

# Benchmark
start = time.perf_counter()
result = scirs2.skew_py(data)
elapsed = time.perf_counter() - start
print(f"Time: {elapsed*1000:.3f} ms")
```

**Solutions**:

1. **Wrong operation** - Use NumPy/SciPy instead:
```python
# ❌ BAD - 377x slower!
det = scirs2.det_py(matrix)

# ✅ GOOD
import scipy.linalg
det = scipy.linalg.det(matrix)
```

2. **Large dataset** - Use NumPy for >10K elements:
```python
if len(data) < 10000:
    skew = scirs2.skew_py(data)  # Fast
else:
    import scipy.stats
    skew = scipy.stats.skew(data)  # Faster for large data
```

3. **Non-contiguous array** - Make contiguous:
```python
# ❌ Slower
data_slice = large_array[::2, ::2]
result = scirs2.skew_py(data_slice.flatten())

# ✅ Faster
data_contiguous = np.ascontiguousarray(data_slice.flatten())
result = scirs2.skew_py(data_contiguous)
```

---

#### Problem: Getting different results than SciPy

**Cause**: Numerical precision differences

**Diagnostic**:
```python
import numpy as np
from scipy import stats
import scirs2

data = np.random.randn(500)

scipy_result = stats.skew(data)
scirs2_result = scirs2.skew_py(data)

diff = abs(scipy_result - scirs2_result)
rel_diff = diff / abs(scipy_result)

print(f"SciPy:  {scipy_result:.15f}")
print(f"scirs2: {scirs2_result:.15f}")
print(f"Absolute diff: {diff:.2e}")
print(f"Relative diff: {rel_diff:.2e}")
```

**Expected**: Relative difference < 1e-10 (numerical precision)

**Solution**: Use tolerance in comparisons:
```python
import numpy as np
assert np.allclose(scipy_result, scirs2_result, rtol=1e-10)
```

**If difference is large** (>1e-6):
- Check input data type (should be float64)
- Verify same parameters (e.g., ddof)
- Report bug at https://github.com/cool-japan/scirs/issues

---

### Issue 3: Type Errors

#### Problem: `TypeError: argument 'data': 'list' object cannot be converted to 'PyArray'`

**Cause**: Passing Python list instead of NumPy array

**Solution**: Convert to NumPy array:
```python
# ❌ Wrong
data = [1, 2, 3, 4, 5]
result = scirs2.mean_py(data)  # Error!

# ✅ Correct
import numpy as np
data = np.array([1, 2, 3, 4, 5])
result = scirs2.mean_py(data)  # Works!
```

---

#### Problem: `TypeError: argument 'data': 'ndarray' object cannot be cast as 'ndarray'`

**Cause**: Wrong array dtype (complex instead of real, or vice versa)

**Diagnostic**:
```python
import numpy as np

data = np.array([1+2j, 3+4j])  # Complex array
print(f"dtype: {data.dtype}")

# Check function requirements
# fft_py requires complex, rfft_py requires real
```

**Solution**: Use correct function for data type:
```python
# Real data
real_data = np.random.randn(1024)
result = scirs2.rfft_py(real_data)  # ✅ Correct

# Complex data (if supported)
complex_data = np.random.randn(1024) + 1j * np.random.randn(1024)
result = scirs2.fft_py(complex_data)  # Check if function exists
```

---

### Issue 4: Memory Issues

#### Problem: `MemoryError` with large arrays

**Cause**: Insufficient memory for operation

**Diagnostic**:
```python
import numpy as np
import psutil

data = np.random.randn(10000000)
print(f"Array size: {data.nbytes / 1024**2:.1f} MB")
print(f"Available memory: {psutil.virtual_memory().available / 1024**2:.1f} MB")
```

**Solutions**:

1. **Process in chunks**:
```python
def process_large_array(data, chunk_size=10000):
    """Process large array in chunks"""
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        results.append(scirs2.skew_py(chunk))
    return np.mean(results)
```

2. **Use streaming approach**:
```python
# Read data in batches
def analyze_large_dataset(filename):
    # Load data incrementally
    stats = []
    with open(filename) as f:
        for chunk in pd.read_csv(f, chunksize=10000):
            data = chunk.values.flatten()
            stats.append(scirs2.skew_py(data))
    return stats
```

3. **Switch to NumPy** (more memory-efficient for some operations):
```python
# Use NumPy for very large data
large_data = np.random.randn(10000000)
mean = np.mean(large_data)  # More efficient than scirs2 for large data
```

---

### Issue 5: Installation Issues

#### Problem: `error: failed to run custom build command for 'scirs2'`

**Cause**: Rust not installed or outdated

**Solution**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust
rustup update

# Install maturin
pip install maturin

# Build
cd scirs2-python
maturin develop --release
```

**Verify Rust installation**:
```bash
rustc --version
cargo --version
```

---

#### Problem: `ImportError: undefined symbol` (Linux)

**Cause**: Missing system dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel

# Rebuild
pip uninstall scirs2
pip install scirs2 --no-cache-dir
```

---

#### Problem: `Library not loaded` (macOS)

**Cause**: Incompatible architecture (Intel vs Apple Silicon)

**Diagnostic**:
```bash
# Check Python architecture
python -c "import platform; print(platform.machine())"

# Check library architecture
file $(python -c "import scirs2; print(scirs2.__file__)")
```

**Solution**: Install correct architecture:
```bash
# For Apple Silicon (M1/M2/M3)
arch -arm64 pip install scirs2

# For Intel
arch -x86_64 pip install scirs2
```

---

### Issue 6: Unexpected Behavior

#### Problem: `describe_py` returns dict instead of tuple

**Cause**: API difference from SciPy

**SciPy**:
```python
from scipy import stats
result = stats.describe(data)
print(result.mean)  # namedtuple
```

**scirs2**:
```python
import scirs2
result = scirs2.describe_py(data)
print(result['mean'])  # dict
```

**Solution**: Update code to use dict syntax:
```python
# ❌ Old (SciPy style)
mean = result.mean

# ✅ New (scirs2 style)
mean = result['mean']
```

---

#### Problem: `pearsonr_py` doesn't return p-value

**Cause**: scirs2 returns correlation only

**SciPy**:
```python
from scipy import stats
corr, pval = stats.pearsonr(x, y)
```

**scirs2**:
```python
import scirs2
corr = scirs2.pearsonr_py(x, y)  # No p-value
```

**Solution**: Use SciPy if you need p-values:
```python
from scipy import stats
import scirs2

# Get correlation from scirs2 (faster)
corr = scirs2.pearsonr_py(x, y)

# Get p-value from SciPy (when needed)
_, pval = stats.pearsonr(x, y)
```

---

## Performance Debugging

### Benchmark Your Code

```python
import time
import numpy as np
from scipy import stats
import scirs2

def benchmark(func, data, n_runs=100):
    """Benchmark a function"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(data)
        times.append(time.perf_counter() - start)
    return np.median(times)

# Test data
data = np.random.randn(1000)

# Benchmark both
scipy_time = benchmark(lambda d: stats.skew(d), data)
scirs2_time = benchmark(lambda d: scirs2.skew_py(d), data)

speedup = scipy_time / scirs2_time

print(f"SciPy time:  {scipy_time*1000:.3f} ms")
print(f"scirs2 time: {scirs2_time*1000:.3f} ms")
print(f"Speedup:     {speedup:.2f}x")

if speedup < 1.0:
    print(f"⚠️  scirs2 is SLOWER - consider using SciPy")
else:
    print(f"✅ scirs2 is faster!")
```

### Profile Your Application

```python
import cProfile
import pstats
import numpy as np
import scirs2

def analyze_data():
    """Function to profile"""
    data = np.random.randn(1000)
    skew = scirs2.skew_py(data)
    kurt = scirs2.kurtosis_py(data)
    return skew, kurt

# Profile
profiler = cProfile.Profile()
profiler.enable()

for _ in range(100):
    analyze_data()

profiler.disable()

# Show results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Getting Help

### Before Reporting Issues

1. **Check documentation**:
   - [Performance Guide](../performance.md)
   - [Quick Start](./quickstart.md)
   - [Migration Guide](./migration.md)

2. **Search existing issues**:
   - https://github.com/cool-japan/scirs/issues

3. **Verify installation**:
   ```python
   import scirs2
   print(scirs2.__version__)
   ```

4. **Create minimal example**:
   ```python
   import numpy as np
   import scirs2

   # Minimal code that reproduces the issue
   data = np.random.randn(100)
   result = scirs2.skew_py(data)
   print(result)
   ```

### Reporting Bugs

Include this information:
1. **scirs2 version**: `scirs2.__version__`
2. **Python version**: `python --version`
3. **NumPy version**: `np.__version__`
4. **Operating system**: Linux/macOS/Windows + version
5. **Minimal reproducing example**
6. **Expected vs actual behavior**
7. **Error message** (full traceback)

**Example bug report**:
```
Title: skew_py raises TypeError with 2D array

Environment:
- scirs2: 0.3.1
- Python: 3.11.5
- NumPy: 1.24.3
- OS: macOS 14.0 (Apple Silicon)

Code:
```python
import numpy as np
import scirs2

data = np.random.randn(10, 10)
result = scirs2.skew_py(data)  # Error!
```

Error:
```
TypeError: argument 'data': expected 1-D array, got 2-D array
```

Expected: Should work or provide clear error message about 1-D requirement
Actual: Confusing error message
```

Submit at: https://github.com/cool-japan/scirs/issues

---

## FAQ

### Q: Why is scirs2 slower than NumPy for basic operations?

**A**: Python binding overhead affects all operations. NumPy has decades of optimization. Use scirs2 only for complex statistics where it's faster (skewness, kurtosis).

See: [Performance Guide](../performance.md)

---

### Q: Can I use scirs2 for production?

**A**: Yes, but **only for complex statistics** on small-medium data. **Do NOT use** for linear algebra, FFT, or large datasets. Use hybrid approach with NumPy/SciPy.

See: [Migration Guide](./migration.md#migration-decision-tree)

---

### Q: Why doesn't scirs2 have p-values for statistical tests?

**A**: Current version focuses on descriptive statistics. Use SciPy for inferential statistics (hypothesis tests, p-values).

---

### Q: How do I contribute?

**A**: See CONTRIBUTING.md in the main repository:
https://github.com/cool-japan/scirs

---

### Q: Is scirs2 compatible with pandas?

**A**: Yes, convert pandas Series/DataFrame to NumPy array:
```python
import pandas as pd
import scirs2

df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
data = df['col'].values  # Convert to NumPy array
skew = scirs2.skew_py(data)
```

---

### Q: Can I use scirs2 in Jupyter notebooks?

**A**: Yes, works perfectly:
```python
import numpy as np
import scirs2

data = np.random.randn(500)
skew = scirs2.skew_py(data)
print(f"Skewness: {skew:.4f}")
```

---

### Q: Does scirs2 support GPU acceleration?

**A**: No, current version is CPU-only. GPU support may be added in future versions.

---

### Q: How do I update scirs2?

**A**:
```bash
pip install --upgrade scirs2
```

Verify:
```python
import scirs2
print(scirs2.__version__)  # Should show latest version
```

---

## Quick Reference

### Checklist for Common Issues

- [ ] scirs2 installed: `pip install scirs2`
- [ ] NumPy installed: `pip install numpy`
- [ ] Data is NumPy array: `data = np.array(...)`
- [ ] Array is contiguous: `data = np.ascontiguousarray(data)`
- [ ] Using correct function for operation:
  - ✅ Complex stats (skew/kurt) on small data
  - ❌ Linear algebra (use SciPy)
  - ❌ FFT (use NumPy)
- [ ] Data size < 10,000 for best performance
- [ ] Compared results with SciPy: `np.allclose(scipy_result, scirs2_result)`
- [ ] Benchmarked performance for your use case

---

## Still Having Issues?

1. **Documentation**: https://docs.rs/scirs2-python
2. **GitHub Issues**: https://github.com/cool-japan/scirs/issues
3. **Performance Questions**: See [Performance Guide](../performance.md)

---

**Last Updated**: 2025-12-29
**Version**: 0.2.0
