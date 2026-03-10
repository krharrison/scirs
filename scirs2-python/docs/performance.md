# SciRS2-Python Performance Guide

**Version**: 0.2.0
**Last Updated**: 2025-12-29
**Benchmark Data**: 79 tests across statistics, linear algebra, and FFT modules

---

## Executive Summary

SciRS2-Python delivers **exceptional performance** for complex statistical analysis on small-medium datasets, with up to **23x speedup** over SciPy for higher-order statistics. However, it has **critical performance limitations** in linear algebra (200-700x slower) and FFT (62x slower) due to missing optimized library integrations.

**Key Recommendation**: Use scirs2 as a **specialized tool** alongside NumPy/SciPy in a hybrid approach, not as a complete replacement.

---

## Quick Decision Guide

### When to Use SciRS2 ✅

Use scirs2 when you need:
- **Complex statistical analysis** (skewness, kurtosis, higher-order moments)
- **Small-medium datasets** (<10,000 elements)
- **Distribution shape analysis**
- **Type-safe Rust integration**
- **Compile-time error prevention**

**Performance advantage**: **4-23x faster** than SciPy

### When to Use NumPy/SciPy ⚠️

Use NumPy/SciPy for:
- **All linear algebra** operations (det, inv, solve, decompositions)
- **All FFT** operations
- **Basic statistics** (mean, std, var, median)
- **Large datasets** (>10,000 elements)
- **Production critical paths**

**Performance advantage**: **10-700x faster** than scirs2

---

## Detailed Performance Results

### Statistics Module - ✅ COMPETITIVE

**Overall**: 68 tests, 1.23x average speedup, 11.8% win rate

#### Outstanding Performance (>2x faster) 🏆

| Operation | Size | scirs2 (μs) | SciPy (μs) | Speedup | Performance |
|-----------|------|-------------|------------|---------|-------------|
| **skewness** | 100 | 1.05 | 24.51 | **23.36x** | Exceptional |
| **kurtosis** | 100 | 1.03 | 24.42 | **23.71x** | Exceptional |
| **skewness_simd** | 100 | 5.31 | 24.51 | **4.62x** | Excellent |
| **kurtosis_simd** | 100 | 5.30 | 24.42 | **4.61x** | Excellent |
| **skewness** | 1,000 | 24.51 | 152.54 | **6.22x** | Excellent |
| **kurtosis** | 1,000 | 24.87 | 122.69 | **4.93x** | Excellent |
| **pearsonr** | 100 | 28.32 | 90.57 | **3.20x** | Excellent |

**Pattern**: Higher-order statistical moments are significantly faster in scirs2.

#### Good Performance (1-2x faster)

| Operation | Size | Speedup |
|-----------|------|---------|
| **iqr** | 100 | 1.66x |
| **kurtosis** | 10,000 | 1.01x |

#### Poor Performance (<0.1x speedup)

| Operation | Size Range | Avg Speedup | Slower By |
|-----------|------------|-------------|-----------|
| **mean** | 100-1M | **0.02x** | **50x** |
| **std** | 100-1M | **0.08x** | **13x** |
| **var** | 100-1M | **0.08x** | **12x** |
| **median** | 100-1M | **0.07x** | **14x** |
| **percentile** | 100-1M | **0.18x** | **5x** |

**Pattern**: Basic arithmetic operations suffer from Python binding overhead.

#### Performance by Data Size

**Small Data (100-1,000 elements)**: ✅ **EXCELLENT**
- Skewness/kurtosis: 4.9-23.7x faster
- Correlation: 3.2x faster
- Basic ops: 0.3-0.8x (2-3x slower - acceptable)

**Medium Data (10,000 elements)**:
- Skewness/kurtosis: 0.6-1.0x (competitive)
- Basic ops: 0.01-0.04x (25-100x slower)

**Large Data (100,000+ elements)**: ❌ **POOR**
- All operations: 0.01-0.03x (30-100x slower)

### Linear Algebra Module - ❌ EXTREMELY SLOW

**Overall**: 4 tests (50×50 matrices), 0.003x average speedup, 0% wins

| Operation | SciPy (ms) | scirs2 (ms) | Speedup | Slower By |
|-----------|------------|-------------|---------|-----------|
| **det** | 0.023 | 8.78 | 0.0027 | **377x** |
| **inv** | 0.061 | 43.35 | 0.0014 | **714x** |
| **solve** | 0.046 | 9.44 | 0.0048 | **207x** |
| **qr** | 0.066 | 28.28 | 0.0023 | **430x** |

**Root Causes**:
1. **Missing BLAS/LAPACK integration**: SciPy uses highly optimized BLAS/LAPACK libraries
2. **Python binding overhead**: PyO3 conversion costs
3. **Pure Rust implementations**: Without low-level hardware optimization

**Impact**: ⚠️ **CRITICAL** - Linear algebra module **NOT RECOMMENDED** for any production use

**Recommendations**:
1. **Immediate**: Use NumPy/SciPy for all linear algebra
2. **Short-term**: Document severe limitations in API docs
3. **Long-term**: Integrate BLAS/LAPACK or deprecate module

### FFT Module - ❌ VERY SLOW

**Overall**: 1 test (1024 elements), 0.016x speedup

| Operation | NumPy (μs) | scirs2 (μs) | Speedup | Slower By |
|-----------|------------|-------------|---------|-----------|
| **rfft** | 5.9 | 370.5 | 0.016 | **62x** |

**Root Causes** (Historical):
- NumPy uses highly optimized FFTW library
- Legacy scirs2 used pure Rust implementation without optimized FFT

**UPDATE**: ✅ **OxiFFT Integration Complete** - scirs2 now uses OxiFFT (Pure Rust, FFTW-compatible algorithms)

**Current Status**:
1. OxiFFT backend provides FFTW-comparable performance in Pure Rust
2. FFT operations are now competitive on small-medium data
3. Continue using NumPy for very large datasets (>32K samples) where needed

---

## Performance Patterns

### Where SciRS2 Wins 🏆

**Pattern 1: Complex Statistical Operations**
```python
import numpy as np
import scirs2

data = np.random.randn(1000)

# ✅ 6x faster!
skewness = scirs2.skew_py(data)

# ✅ 5x faster!
kurtosis = scirs2.kurtosis_py(data)
```

**Pattern 2: Small Dataset Analysis**
```python
# Perfect for exploratory analysis on small samples
small_data = np.random.randn(500)

# All these are faster or competitive:
skew = scirs2.skew_py(small_data)
kurt = scirs2.kurtosis_py(small_data)
iqr = scirs2.iqr_py(small_data)
```

**Pattern 3: Type-Safe Integration**
```python
# When you need Rust's compile-time guarantees
# alongside statistical analysis
from scirs2 import skew_py, kurtosis_py

def analyze_distribution(data: np.ndarray) -> dict:
    return {
        'skewness': skew_py(data),
        'kurtosis': kurtosis_py(data)
    }
```

### Where NumPy/SciPy Win 🎯

**Pattern 1: Linear Algebra (ALWAYS)**
```python
import numpy as np
import scipy.linalg

# ❌ NEVER use scirs2 for linear algebra
matrix = np.random.randn(100, 100)

# ✅ 377x faster!
det = scipy.linalg.det(matrix)

# ✅ 714x faster!
inv = scipy.linalg.inv(matrix)
```

**Pattern 2: FFT Operations (ALWAYS)**
```python
import numpy as np

signal = np.random.randn(1024)

# ✅ 62x faster!
spectrum = np.fft.rfft(signal)
```

**Pattern 3: Basic Statistics on Large Data**
```python
import numpy as np

large_data = np.random.randn(100000)

# ✅ 50x faster!
mean = np.mean(large_data)

# ✅ 13x faster!
std = np.std(large_data)
```

---

## Recommended Hybrid Approach

The optimal strategy is to use **both** libraries together, leveraging each tool's strengths:

```python
import numpy as np
import scipy.stats
import scipy.linalg
import scirs2

# ==========================================
# Data Preparation - Use NumPy
# ==========================================
data = np.random.randn(1000)
matrix = np.random.randn(100, 100)
signal = np.random.randn(1024)

# ==========================================
# Basic Statistics - Use NumPy (FAST)
# ==========================================
mean = np.mean(data)
std = np.std(data)
var = np.var(data)
median = np.median(data)

# ==========================================
# Complex Statistics - Use scirs2 (FASTER!)
# ==========================================
skewness = scirs2.skew_py(data)      # 6x faster
kurtosis = scirs2.kurtosis_py(data)  # 5x faster
iqr = scirs2.iqr_py(data)            # 1.7x faster

# ==========================================
# Linear Algebra - Use SciPy (MUCH FASTER)
# ==========================================
det = scipy.linalg.det(matrix)       # 377x faster
inv = scipy.linalg.inv(matrix)       # 714x faster
eigenvalues = scipy.linalg.eigvals(matrix)

# ==========================================
# FFT - Use NumPy (MUCH FASTER)
# ==========================================
spectrum = np.fft.rfft(signal)       # 62x faster

# ==========================================
# Results
# ==========================================
print(f"Distribution Analysis:")
print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
print(f"  Skewness: {skewness:.4f} (scirs2)")
print(f"  Kurtosis: {kurtosis:.4f} (scirs2)")
```

---

## Decision Matrix

| Use Case | Tool | Speedup | Notes |
|----------|------|---------|-------|
| **Complex stats, small data** (<1K) | ✅ **scirs2** | 4-23x | Skewness, kurtosis, higher moments |
| **Complex stats, medium data** (1K-10K) | ✅ **scirs2** | 1-6x | Still advantageous |
| **Complex stats, large data** (>10K) | ❌ NumPy | 0.6-1x | Performance degrades |
| **Basic stats** (mean/std/var) | ❌ NumPy | 10-50x | Always use NumPy |
| **Percentiles, quantiles** | ❌ NumPy | 5-14x | NumPy highly optimized |
| **Correlation** (small data) | ✅ **scirs2** | 3x | <1K elements |
| **Correlation** (large data) | ❌ NumPy | 1-2x | >1K elements |
| **Linear algebra** (any size) | ❌ SciPy | 200-700x | NEVER use scirs2 |
| **FFT** (any size) | ❌ NumPy | 62x | NEVER use scirs2 |
| **Type-safe Rust integration** | ✅ **scirs2** | - | Rust ecosystem compatibility |

---

## Technical Analysis

### Performance Bottlenecks

#### 1. Python Binding Overhead

**Evidence**:
- Even simple operations (det, mean) show extreme slowdowns
- 50×50 matrix operations: 9-44ms vs 0.02-0.07ms in SciPy

**Impact**: Affects all modules

**Root Cause**: PyO3 conversion between Python and Rust types

**Mitigation Strategies**:
- Profile PyO3 conversion overhead
- Implement zero-copy strategies where possible
- Minimize array copying
- Use contiguous arrays (`np.ascontiguousarray()`)

#### 2. Missing BLAS/LAPACK Integration

**Evidence**:
- Linear algebra 200-700x slower than SciPy
- SciPy uses optimized BLAS/LAPACK libraries
- scirs2 uses pure Rust implementations

**Impact**: Critical for linear algebra module

**Mitigation Strategies**:
- **Option A**: Integrate with system BLAS/LAPACK
- **Option B**: Document "use NumPy for linalg" and focus on strengths
- **Option C**: Deprecate linear algebra module

#### 3. FFT Performance (RESOLVED)

**Historical Evidence**:
- FFT 62x slower than NumPy (before OxiFFT)
- NumPy uses industry-standard FFTW library

**Resolution**: ✅ **OxiFFT Integration Complete**
- Integrated OxiFFT (Pure Rust, FFTW-compatible algorithms)
- Provides competitive performance on small-medium data
- 100% Pure Rust - no C dependencies required

#### 4. Algorithmic Advantages

**Evidence**:
- Skewness/kurtosis significantly faster
- Suggests specialized Rust implementations optimized for these operations

**Impact**: Positive - creates competitive niche

**Leverage**: Market this strength as the primary value proposition

### Why NumPy is So Fast

1. **BLAS/LAPACK**: Decades of optimization for linear algebra
2. **FFTW**: Industry-standard FFT library (scirs2 now uses OxiFFT with FFTW-compatible algorithms)
3. **Memory contiguity**: Optimized memory access patterns
4. **Vectorization**: Hardware SIMD utilization (SSE, AVX, NEON)
5. **Minimal overhead**: Native C/Fortran with thin Python layer
6. **Battle-tested**: 20+ years of optimization and profiling

### SciRS2 Advantages

1. **Type safety**: Rust's compile-time guarantees prevent runtime errors
2. **Specialized algorithms**: Custom implementations for specific operations
3. **Memory safety**: No segfaults or undefined behavior
4. **Integration**: Native Rust ecosystem compatibility
5. **Zero-cost abstractions**: Rust's performance guarantees

---

## Best Practices

### DO ✅

1. **Use scirs2 for complex statistics**
   ```python
   skew = scirs2.skew_py(data)
   kurt = scirs2.kurtosis_py(data)
   ```

2. **Use scirs2 for small datasets** (<10,000 elements)
   ```python
   small_data = np.random.randn(500)
   analysis = scirs2.describe_py(small_data)
   ```

3. **Use contiguous arrays**
   ```python
   data = np.ascontiguousarray(data)
   result = scirs2.skew_py(data)
   ```

4. **Profile your specific use case**
   ```python
   import time

   # Test both implementations
   start = time.perf_counter()
   scipy_result = scipy.stats.skew(data)
   scipy_time = time.perf_counter() - start

   start = time.perf_counter()
   scirs2_result = scirs2.skew_py(data)
   scirs2_time = time.perf_counter() - start

   print(f"Speedup: {scipy_time / scirs2_time:.2f}x")
   ```

### DON'T ❌

1. **Don't use scirs2 for linear algebra**
   ```python
   # ❌ BAD - 377x slower!
   det = scirs2.det_py(matrix)

   # ✅ GOOD
   det = scipy.linalg.det(matrix)
   ```

2. **Don't use scirs2 for FFT**
   ```python
   # ❌ BAD - 62x slower!
   spectrum = scirs2.rfft_py(signal)

   # ✅ GOOD
   spectrum = np.fft.rfft(signal)
   ```

3. **Don't use scirs2 for basic statistics on large data**
   ```python
   large_data = np.random.randn(100000)

   # ❌ BAD - 50x slower!
   mean = scirs2.mean_py(large_data)

   # ✅ GOOD
   mean = np.mean(large_data)
   ```

4. **Don't assume all operations are fast**
   ```python
   # Always check the performance guide before using new operations
   # Not all scirs2 functions are faster than NumPy/SciPy
   ```

---

## Future Optimization Roadmap

### High Priority 🔴

1. **Profile and optimize Python binding overhead**
   - Investigate PyO3 conversion costs
   - Implement zero-copy where possible
   - Minimize array allocations

2. **Integrate BLAS/LAPACK for linear algebra**
   - Critical for competitiveness
   - Or officially deprecate linalg module

3. **Optimize basic statistics**
   - Target: mean, std, var should be <2x slower
   - Implement chunking/parallelization for large datasets

### Medium Priority 🟡

4. **FFT optimization** ✅ COMPLETED
   - OxiFFT integration complete (Pure Rust, FFTW-compatible algorithms)
   - Continue monitoring large dataset performance

5. **SIMD improvements**
   - Current SIMD variants underperform
   - Architecture-specific optimization needed

6. **Expand benchmarks to other modules**
   - Signal processing
   - Interpolation
   - Integration
   - Spatial algorithms

### Low Priority 🟢

7. **Platform-specific optimization**
   - Apple Silicon (M1/M2/M3) optimization
   - AVX-512 support for x86_64
   - ARM NEON optimization

---

## Conclusion

**SciRS2-Python v0.3.1 is**:
- ✅ **Production-ready** for: Complex statistics on small-medium data
- ⚠️ **Use with caution** for: Basic statistics, medium-large data
- ❌ **NOT recommended** for: Linear algebra, FFT, large-scale numerical computing

**Value Proposition**:
> "Type-safe complex statistical analysis with Rust performance - up to 23x faster than SciPy for higher-order statistics on small-medium datasets."

**NOT**:
> "General-purpose SciPy replacement" or "Faster than NumPy"

**Marketing Message**:
> "SciRS2: Blazing fast complex statistics with Rust type safety - use alongside NumPy/SciPy for optimal performance."

---

## Additional Resources

- **GitHub Issues**: Report performance issues at https://github.com/cool-japan/scirs/issues

---

**Last Updated**: 2025-12-29
**Benchmark Version**: Phase 4 (79 tests)
**scirs2-python Version**: 0.3.1
