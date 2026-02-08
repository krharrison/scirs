# BLAS Backend Configuration

> **Note (v0.1.5+)**: SciRS2 now uses **OxiBLAS** - a Pure Rust BLAS/LAPACK implementation.
> No system dependencies (OpenBLAS, MKL, Accelerate) are required.

## Overview

As of v0.1.5, SciRS2 has transitioned to [OxiBLAS](https://github.com/cool-japan/oxiblas), a pure Rust implementation of BLAS and LAPACK operations. This eliminates all system dependency requirements and provides consistent behavior across all platforms.

## Benefits of OxiBLAS

- ✅ **Zero System Dependencies** - No need to install OpenBLAS, Intel MKL, or configure Accelerate
- ✅ **Cross-Platform Consistency** - Same code, same behavior on macOS, Linux, and Windows
- ✅ **Simple Installation** - Just `cargo build`, no linker configuration needed
- ✅ **Memory Safety** - Full Rust implementation with compile-time guarantees
- ✅ **Competitive Performance** - Optimized algorithms with SIMD support

## Platform Behavior (v0.1.5+)

**All platforms use OxiBLAS by default**:
- **macOS**: Pure Rust OxiBLAS (no Accelerate dependency)
- **Linux**: Pure Rust OxiBLAS (no OpenBLAS/MKL dependency)
- **Windows**: Pure Rust OxiBLAS (no vcpkg/system BLAS needed)

## Usage

### Default (Recommended)
```bash
# Just build - no special configuration needed
cargo build
cargo test
```

### Enable SIMD Acceleration
```bash
# Enable SIMD for additional performance
cargo build --features simd
```

### Enable Parallel Processing
```bash
# Enable multi-threaded operations
cargo build --features parallel
```

## Performance

OxiBLAS provides competitive performance for most use cases:

| Operation | Size | OxiBLAS | Notes |
|-----------|------|---------|-------|
| Matrix Multiply | 1000×1000 | ~200ms | SIMD-optimized |
| SVD | 1000×1000 | ~150ms | Pure Rust |
| LU Decomposition | 1000×1000 | ~50ms | Parallel support |
| Eigenvalues | 1000×1000 | ~120ms | Pure Rust |

**Note**: Enable `simd` and `parallel` features for best performance.

## Migration from Previous Versions

If you were using system BLAS backends (OpenBLAS, MKL, Accelerate) in earlier versions:

1. **Remove system dependencies** - No longer needed
2. **Remove feature flags** - `openblas`, `netlib`, `accelerate`, `intel-mkl` features are removed
3. **Clean and rebuild**:
   ```bash
   cargo clean
   cargo build
   ```

## Troubleshooting

### Build Issues
If you encounter build issues after upgrading:

1. **Clean the build cache**:
   ```bash
   cargo clean
   cargo build
   ```

2. **Update dependencies**:
   ```bash
   cargo update
   ```

### Performance Concerns
If performance is critical and you need maximum speed:

1. Enable SIMD: `--features simd`
2. Enable parallel processing: `--features parallel`
3. Use release builds: `cargo build --release`

## Historical Note

Previous versions of SciRS2 supported multiple BLAS backends:
- ~~`openblas`~~ - Removed in v0.1.5
- ~~`netlib`~~ - Removed in v0.1.5
- ~~`accelerate`~~ - Removed in v0.1.5
- ~~`intel-mkl`~~ - Removed in v0.1.5

All linear algebra operations now use OxiBLAS, providing a unified, dependency-free experience.