# SciRS2-Python TODO (For v0.2.0 or later)

## Status: NOT READY FOR v0.1.5

This crate contains the foundational work for unified Python bindings but has critical blockers preventing immediate release. It will be included in a future version (v0.2.0 or v0.3.0).

---

## Critical Blockers

### 🔴 **BLOCKER #1: ndarray Version Conflict**

**Problem**: Type mismatch between internal implementations and Python bindings

```
Internal SciRS2 crates (scirs2-series, scirs2-cluster)
    ↓ Use: ndarray 0.17

Python bindings (this crate)
    ↓ Use: ndarray 0.16 (for numpy compatibility)

→ TYPE MISMATCH: Cannot call scirs2_series::auto_arima() from Python bindings
```

**Current Errors**: 107 compilation errors due to type incompatibility

**Resolution Options**:

1. **Type Conversion Layer** (RECOMMENDED)
   - Create `scirs2_series::python_api` module with ndarray16-compatible wrappers
   - Convert at boundary: ndarray16 → ndarray17 → process → ndarray17 → ndarray16
   - Example:
     ```rust
     // In scirs2-series/src/python_api.rs
     pub fn auto_arima_py(data: &Array1<f64>) -> Result<(ArimaModel, ArimaParams)> {
         // Convert ndarray16 → ndarray17
         let data17 = convert_to_ndarray17(data)?;
         // Call internal function
         let result = crate::arima_models::auto_arima(&data17, &options)?;
         // Return (ndarray17 stays internal)
         Ok(result)
     }
     ```

2. **Dual-Version Support in Subcrates**
   - Make scirs2-series/cluster support both ndarray16 and ndarray17 via feature flags
   - Complexity: HIGH (requires duplicating logic)
   - Not recommended

3. **Wait for numpy crate ndarray 0.17 support**
   - Track: https://github.com/PyO3/rust-numpy/issues
   - Status: Unknown timeline
   - Risk: Could take months/years

### 🔴 **BLOCKER #2: Missing Type Conversion Utilities**

**Needed Functions** (not yet implemented in scirs2-core):
- `convert_ndarray16_to_17()` - Convert Array1/Array2 from v0.16 to v0.17
- `convert_ndarray17_to_16()` - Convert Array1/Array2 from v0.17 to v0.16

**Implementation Location**: `scirs2-core/src/python/conversions.rs`

---

## TODO List (Prioritized)

### Phase 0: Preparation (Before starting Python bindings)

- [x] **Implement ndarray conversion utilities in scirs2-core**
  - [x] `convert_array1_16_to_17<T>(array: ndarray16::Array1<T>) -> ndarray17::Array1<T>`
  - [x] `convert_array2_16_to_17<T>(array: ndarray16::Array2<T>) -> ndarray17::Array2<T>`
  - [x] `convert_array1_17_to_16<T>(array: ndarray17::Array1<T>) -> ndarray16::Array1<T>`
  - [x] `convert_array2_17_to_16<T>(array: ndarray17::Array2<T>) -> ndarray16::Array2<T>`
  - [x] Add comprehensive tests for conversions (8 tests passing)
  - [ ] Benchmark conversion overhead

- [x] **Create Python-compatible API wrappers in subcrates**
  - [x] `scirs2-series/src/python_api.rs` - Wrappers for ARIMA, differencing, STL, Box-Cox (4 tests passing)
  - [x] `scirs2-cluster/src/python_api.rs` - Wrappers for K-means, metrics, preprocessing (3 tests passing)
  - [x] Each wrapper accepts/returns ndarray16 types via scirs2-core conversions

### Phase 1: Fix Existing Bindings

- [x] **Fix scirs2-python/src/series.rs** (385 lines)
  - [x] Update `PyTimeSeries` to use conversion layer
  - [x] Update `PyARIMA` to use conversion layer
  - [x] Fix all type mismatch errors
  - [x] Zero warnings, clippy clean

- [x] **Fix scirs2-python/src/cluster.rs** (302 lines)
  - [x] Update `PyKMeans` to use conversion layer
  - [x] Add metric functions (silhouette, davies-bouldin, calinski-harabasz)
  - [x] Add preprocessing functions (standardize, normalize)
  - [x] Fix all type mismatch errors
  - [x] Zero warnings, clippy clean

- [x] **Remove `#[cfg(feature = "python")]` guards**
  - [x] Already done (this crate is Python-only)

### Phase 2: Add New Modules (v0.2.0 scope)

- [x] **scirs2-python/src/linalg.rs** (376 lines)
  - [x] Basic: det, inv, trace
  - [x] Decompositions: LU, QR, SVD, Cholesky, eigenvalues
  - [x] Solvers: solve, lstsq
  - [x] Norms: matrix_norm, vector_norm, cond, rank
  - [x] 16 functions exposed to Python

- [x] **scirs2-python/src/stats.rs** (141 lines)
  - [x] Descriptive: describe, mean, std, var, percentile, median, iqr
  - [x] Correlation and covariance
  - [x] 9 functions exposed to Python

- [x] **scirs2-python/src/fft.rs** (232 lines)
  - [x] Core FFT: fft, ifft, rfft, irfft
  - [x] DCT: dct, idct
  - [x] Helpers: fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len
  - [x] 11 functions exposed to Python

### Phase 3: Testing & Quality

- [ ] **Python Unit Tests**
  - [ ] Create `scirs2-python/tests/` directory
  - [ ] Write pytest tests for all modules
  - [ ] Comparison tests against SciPy (accuracy validation)
  - [ ] Performance benchmarks

- [ ] **Documentation**
  - [ ] API documentation (docstrings)
  - [ ] User guide
  - [ ] Migration guide from SciPy
  - [ ] Performance comparison tables

- [ ] **CI/CD**
  - [ ] GitHub Actions for maturin build
  - [ ] Test on Linux, macOS, Windows
  - [ ] Automated PyPI publishing
  - [ ] Pre-built wheels for common platforms

### Phase 4: Release (v0.2.0)

- [ ] **Publishing**
  - [ ] Test publish to test.pypi.org
  - [ ] Publish to PyPI
  - [ ] Announce on Reddit r/rust, r/Python, r/MachineLearning

- [ ] **Community**
  - [ ] Add to Awesome Rust list
  - [ ] Submit to Python Weekly
  - [ ] Blog post with benchmarks

---

## Technical Notes

### ndarray Version Strategy

**Internal (Rust)**: ndarray 0.17
- Used by: scirs2-core, scirs2-series, scirs2-cluster, etc.
- Benefits: Latest features, performance improvements

**Python Boundary**: ndarray 0.16
- Used by: scirs2-python, scirs2-core::python module
- Reason: numpy crate only supports ndarray ≤ 0.16
- Conversion: Zero-copy where possible, clone when necessary

### Type Conversion Pattern

```rust
// In scirs2-python/src/series.rs
fn apply_differencing(py: Python, data: &PyTimeSeries, periods: usize)
    -> PyResult<Py<PyArray1<f64>>>
{
    // data.values is ndarray16::Array1<f64>

    // Convert to ndarray17
    let data17 = scirs2_core::python::convert_array1_16_to_17(&data.values)?;

    // Call scirs2-series function (ndarray17)
    let result17 = scirs2_series::difference_series(&data17, periods)?;

    // Convert back to ndarray16
    let result16 = scirs2_core::python::convert_array1_17_to_16(&result17)?;

    // Return to Python
    scirs_to_numpy_array1(result16, py)
}
```

### Build Configuration

**IMPORTANT**: This crate should NOT be included in default workspace builds

```toml
# In workspace Cargo.toml
[workspace]
members = [
    # ... other crates ...
    "scirs2-python",  # ← Commented out for v0.1.5 builds
]
```

### Maturin Commands (For Future)

```bash
# Development build
maturin develop

# Build wheel
maturin build --release

# Publish to test PyPI
maturin publish --repository testpypi

# Publish to PyPI
maturin publish
```

---

## Estimated Effort

| Phase | Task | Effort | Dependencies |
|-------|------|--------|--------------|
| 0 | Conversion utilities | 1 week | None |
| 0 | Python API wrappers | 2 weeks | Conversion utilities |
| 1 | Fix existing bindings | 2 weeks | Python API wrappers |
| 2 | Add new modules | 4-6 weeks | Phase 1 |
| 3 | Testing & docs | 2-3 weeks | Phase 2 |
| 4 | Release prep | 1 week | Phase 3 |
| **TOTAL** | **12-15 weeks** | **3-4 months** | - |

---

## Success Criteria (For v0.2.0 Release)

- [ ] Zero compilation errors
- [ ] Zero clippy warnings
- [ ] All Python tests pass
- [ ] Performance within 2x of SciPy on benchmarks
- [ ] Documentation completeness > 80%
- [ ] Type stubs generated for IDE support
- [ ] Pre-built wheels for Linux/macOS/Windows
- [ ] PyPI listing with good README

---

## Current State (v0.1.5)

✅ **Phase 0 Complete** (2025-12):
- Cargo.toml, pyproject.toml, README.md
- lib.rs with submodule framework
- **scirs2-core/src/python/version_conversions.rs** - ndarray16 ↔ ndarray17 conversions (8 tests passing)
- **scirs2-series/src/python_api.rs** - ARIMA, differencing, STL, Box-Cox wrappers (4 tests passing)
- **scirs2-cluster/src/python_api.rs** - K-means, metrics, preprocessing wrappers (3 tests passing)

✅ **Phase 1 Complete** (2025-12):
- **scirs2-python/src/series.rs** (385 lines) - Uses python_api wrappers, zero warnings
- **scirs2-python/src/cluster.rs** (302 lines) - Uses python_api wrappers, zero warnings
- Build succeeds with zero warnings
- Clippy passes with zero warnings
- **Python bindings tested and working** with Python 3.13

⚠️ **Important**: PyO3 links against the system Python (homebrew Python 3.13 on macOS). Users must use matching Python version.

✅ **Phase 2 Complete** (2025-12):
- **scirs2-linalg/src/python_api.rs** (360 lines) - det, inv, lu, qr, svd, cholesky, eig, solve, norms (5 tests passing)
- **scirs2-python/src/linalg.rs** (376 lines) - Full PyO3 bindings for linalg
- **scirs2-stats/src/python_api.rs** (240 lines) - describe, mean, std, var, percentile, correlation (4 tests passing)
- **scirs2-python/src/stats.rs** (141 lines) - Full PyO3 bindings for stats
- **scirs2-fft/src/python_api.rs** (277 lines) - fft, ifft, rfft, dct, helpers (5 tests passing)
- **scirs2-python/src/fft.rs** (232 lines) - Full PyO3 bindings for fft
- **Total: 1,466 lines of Rust code across 6 modules**
- **50+ Python functions across 5 modules**
- All Python tests pass with Python 3.13

🎯 **Ready for Phase 3**: Testing, documentation, CI/CD for v0.2.0

---

## References

- **PyO3 Guide**: https://pyo3.rs/
- **Maturin**: https://www.maturin.rs/
- **rust-numpy**: https://github.com/PyO3/rust-numpy
- **SciPy API Reference**: https://docs.scipy.org/doc/scipy/reference/
- **scikit-learn API**: https://scikit-learn.org/stable/modules/classes.html

---

## Notes for Future Developers

1. **Don't remove `#[cfg(feature = "python")]` guards from cluster/series.rs** - They were removed as an experiment but should be restored
2. **Type conversions have overhead** - Benchmark critical paths before shipping
3. **NumPy version compatibility** - Test with NumPy 1.20+ and 2.0+
4. **Memory safety** - Be careful with lifetime management in PyO3
5. **Error messages** - Make them user-friendly for Python users (not Rust errors)

---

Last Updated: 2025-12
Status: Phase 0 Complete - Foundation work done
Target: v0.2.0 or v0.3.0
