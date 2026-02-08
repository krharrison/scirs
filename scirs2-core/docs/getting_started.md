# Getting Started with SciRS2-Core

## Introduction

SciRS2-Core is a high-performance scientific computing library for Rust that provides:

- **SIMD-accelerated operations** with up to 14x performance improvements
- **GPU acceleration** support for CUDA, ROCm, Metal, and WebGPU
- **Advanced memory management** with intelligent pooling and allocation strategies
- **Parallel processing** with automatic chunking and load balancing
- **Comprehensive error handling** with location tracking and context chaining
- **Property-based testing** framework for mathematical property verification

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-core = { version = "0.1.5", features = ["simd", "parallel", "gpu"] }
```

### Basic Usage

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    parallel_ops::parallel_map,
    memory::create_scientific_pool,
    testing::NumericAssertion,
};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create high-performance memory pool
    let mut pool = create_scientific_pool::<f64>();

    // Generate data using optimized allocation
    let data = pool.acquire_vec_advanced(10000);

    // SIMD-accelerated operations
    let a = Array1::from_vec((0..1000).map(|x| x as f64).collect());
    let b = Array1::from_vec((0..1000).map(|x| (x * 2) as f64).collect());

    // Element-wise multiplication with SIMD
    let result = f64::simd_mul(&a.view(), &b.view());

    // Parallel processing
    let processed = parallel_map(&result.as_slice().unwrap(), |&x| x.sqrt());

    // Validation with numerical assertions
    for &value in &processed {
        value.assert_finite();
        value.assert_in_range(0.0, f64::MAX);
    }

    println!("Processed {} elements successfully!", processed.len());
    Ok(())
}
```

## Core Concepts

### 1. SIMD Operations

SIMD (Single Instruction, Multiple Data) operations provide significant performance improvements:

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
use ndarray::Array1;

// Traditional scalar operation
let mut result = Array1::zeros(1000);
for i in 0..1000 {
    result[i] = a[i] + b[i];
}

// SIMD-accelerated operation (up to 14x faster)
let result = f64::simd_add(&a.view(), &b.view());
```

**Available SIMD Operations:**
- `simd_add()`: Element-wise addition
- `simd_mul()`: Element-wise multiplication
- `simd_dot()`: Dot product
- `simd_norm()`: L2 norm
- `simd_sum()`: Sum all elements

### 2. Memory Management

Advanced memory management with multiple allocation strategies:

```rust
use scirs2_core::memory::{AdvancedBufferPool, MemoryConfig, AllocationStrategy};

// Create optimized memory pool
let config = MemoryConfig {
    strategy: AllocationStrategy::CacheAligned,
    alignment: 64,
    numa_aware: true,
    ..Default::default()
};

let mut pool = AdvancedBufferPool::<f64>::with_config(config);

// Acquire and release buffers efficiently
let buffer = pool.acquire_vec_advanced(1000);
// ... use buffer
pool.release_vec_advanced(buffer);

// Monitor memory usage
let report = pool.memory_report();
println!("Pool efficiency: {:.1}%", report.pool_efficiency * 100.0);
```

### 3. Parallel Processing

Intelligent parallel processing with automatic chunking:

```rust
use scirs2_core::{parallel_ops::parallel_map, chunking::ChunkConfig};

// Automatic parallel processing
let results = parallel_map(&data, |item| expensive_computation(item));

// Optimized chunking for specific workloads
let chunk_config = ChunkConfig::linear_algebra(); // For matrix operations
let chunk_config = ChunkConfig::monte_carlo();    // For random sampling
let chunk_config = ChunkConfig::gpu_hybrid();     // For CPU/GPU workloads
```

### 4. Error Handling

Comprehensive error handling with context and location tracking:

```rust
use scirs2_core::error::{CoreError, ErrorContext, ErrorLocation};

fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, CoreError> {
    if a.ncols() != b.nrows() {
        return Err(CoreError::DimensionError(
            ErrorContext::new("Matrix dimensions incompatible for multiplication")
                .with_location(ErrorLocation::new(file!(), line!()))
        ));
    }

    Ok(a.dot(b))
}

// Using convenience macros
fn validate_positive(value: f64) -> Result<(), CoreError> {
    if value <= 0.0 {
        return Err(valueerror!("Value must be positive"));
    }
    Ok(())
}
```

### 5. GPU Acceleration

Multi-backend GPU support:

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, ElementwiseAddKernel};

// Initialize GPU context
let context = GpuContext::new(GpuBackend::CUDA, 0)?;

// Create and execute GPU kernel
let kernel = ElementwiseAddKernel::new();
kernel.launch(&context, &[buffer_a, buffer_b, buffer_result])?;
```

## Performance Optimization

### SIMD Performance

The library provides multiple SIMD optimization levels:

```rust
use scirs2_core::simd::{
    simd_mul_f32_lightweight,     // Basic SIMD
    simd_mul_f32_pipelined,       // Software pipelining
    simd_mul_f32_cacheline,       // Cache-optimized
    simd_mul_f32_hyperoptimized,  // Adaptive selection
};

// Adaptive selection based on data size
let result = simd_mul_f32_hyperoptimized(&a.view(), &b.view());
```

**Performance Benchmarks:**
- Lightweight SIMD: 1.30x faster than scalar
- Pipelined SIMD: 3.42x faster than scalar
- Cache-optimized: 8.91x faster than scalar
- Hyperoptimized: up to 14.17x faster than scalar

### Memory Optimization

```rust
use scirs2_core::memory::{MemoryPressure, AccessPattern};

// Check memory pressure and adapt
let pressure = pool.memory_pressure();
match pressure {
    MemoryPressure::High | MemoryPressure::Critical => {
        pool.compact(); // Reduce memory usage
    }
    _ => {} // Normal operation
}

// Set access pattern hints for optimization
let config = MemoryConfig {
    access_pattern: AccessPattern::Sequential, // For linear access
    access_pattern: AccessPattern::Random,     // For random access
    access_pattern: AccessPattern::Streaming,  // For one-pass access
    ..Default::default()
};
```

### Chunking Optimization

```rust
use scirs2_core::chunking::{ChunkConfig, ChunkingUtils};

// Automatic optimal chunking
let config = ChunkConfig::default();
let chunk_size = ChunkingUtils::optimal_chunk_size(data.len(), &config);

// Workload-specific optimizations
let config = ChunkConfig::linear_algebra()      // Matrix operations
    .with_monitoring()                          // Enable performance tracking
    .with_numa_strategy(NumaStrategy::LocalPreferred);

let results = ChunkingUtils::chunked_map(&data, &config, |chunk| {
    process_chunk(chunk)
});
```

## Testing and Validation

### Numerical Testing

```rust
use scirs2_core::testing::{NumericAssertion, TestDataGenerator, PropertyTest};

// Precise numerical assertions
let result = compute_result();
result.assert_approx_eq(&expected, 1e-12);     // Absolute tolerance
result.assert_relative_eq(&expected, 1e-10);   // Relative tolerance

// Generate test data
let mut gen = TestDataGenerator::with_seed(42);
let matrix = gen.positive_definite_matrix(100);
let sparse = gen.sparse_matrix((1000, 1000), 0.95, -1.0, 1.0);

// Property-based testing
PropertyTest::new()
    .with_iterations(1000)
    .test_property("matrix_multiply_associative", |gen| {
        let a = gen.random_matrix((10, 10), -1.0, 1.0);
        let b = gen.random_matrix((10, 10), -1.0, 1.0);
        let c = gen.random_matrix((10, 10), -1.0, 1.0);

        let left = (a.dot(&b)).dot(&c);
        let right = a.dot(&(b.dot(&c)));

        left.assert_approx_eq(&right, 1e-12);
    });
```

### Performance Testing

```rust
use scirs2_core::testing::BenchmarkSuite;
use std::time::Instant;

let mut suite = BenchmarkSuite::new()
    .with_regression_threshold(1.1); // 10% regression detection

suite.add_benchmark("simd_multiplication", || {
    let start = Instant::now();
    let result = f64::simd_mul(&a.view(), &b.view());
    start.elapsed()
});

let results = suite.run_all();
results.print_results();
```

## Mathematical Constants

Access comprehensive mathematical and physical constants:

```rust
use scirs2_core::constants::{math, physical, numerical, complex};

// Mathematical constants
let area = math::PI * radius.powi(2);
let golden_ratio = math::GOLDEN_RATIO;
let catalan = math::CATALAN;

// Physical constants
let energy = physical::PLANCK * frequency;
let force = mass * physical::STANDARD_GRAVITY;

// Numerical constants
let tolerance = numerical::DEFAULT_TOLERANCE;
let epsilon = numerical::MACHINE_EPSILON_F64;

// Complex constants
let imaginary_unit = complex::I;
let euler_identity = complex::E_TO_I_PI; // e^(iπ) = -1
```

## Common Patterns

### Scientific Computation Pipeline

```rust
use scirs2_core::{
    memory::create_scientific_pool,
    chunking::ChunkConfig,
    parallel_ops::parallel_map,
    simd_ops::SimdUnifiedOps,
    validation::{ProductionValidator, ValidationContext},
    testing::NumericAssertion,
};

fn scientific_pipeline(data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // 1. Setup optimized memory management
    let mut pool = create_scientific_pool::<f64>();

    // 2. Validate input data
    let validator = ProductionValidator::new();
    let context = ValidationContext::production();
    validator.validate_with_context(&data, &context)?;

    // 3. Parallel processing with optimized chunking
    let chunk_config = ChunkConfig::compute_intensive();
    let intermediate = parallel_map(data, |&value| {
        // SIMD-accelerated computation
        value.sqrt() * std::f64::consts::PI
    });

    // 4. Further processing with SIMD
    let array = Array1::from_vec(intermediate);
    let normalized = {
        let norm = f64::simd_norm(&array.view());
        if norm > 0.0 {
            f64::simd_mul(&array.view(), &Array1::from_elem(array.len(), 1.0 / norm).view())
        } else {
            array
        }
    };

    // 5. Validation of results
    normalized.assert_finite();

    // 6. Return results
    Ok(normalized.to_vec())
}
```

### GPU-Accelerated Computation

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, GpuBuffer};

fn gpu_computation(data: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = GpuContext::new(GpuBackend::CUDA, 0)?;

    // Create GPU buffers
    let input_buffer = GpuBuffer::from_slice(&context, data)?;
    let output_buffer = GpuBuffer::zeros(&context, data.len())?;

    // Execute GPU kernel
    let kernel = ElementwiseSquareKernel::new();
    kernel.launch(&context, &[&input_buffer, &output_buffer])?;

    // Copy results back to host
    let result = output_buffer.to_vec()?;

    Ok(result)
}
```

## Best Practices

### 1. Memory Management

- Use `create_scientific_pool()` for general scientific computing
- Use `create_large_data_pool()` for datasets larger than memory
- Monitor memory pressure with `memory_report()`
- Call `compact()` during high memory pressure

### 2. Performance Optimization

- Use SIMD operations for array computations
- Choose appropriate chunking strategies for your workload
- Enable NUMA awareness for multi-socket systems
- Profile with `BenchmarkSuite` to detect regressions

### 3. Error Handling

- Use specific error types (`DomainError`, `ValueError`, etc.)
- Include location information with error context
- Chain errors to preserve causation information
- Validate inputs at function boundaries

### 4. Testing

- Use `NumericAssertion` for floating-point comparisons
- Generate test data with `TestDataGenerator`
- Test mathematical properties with `PropertyTest`
- Benchmark performance with `BenchmarkSuite`

### 5. GPU Computing

- Check GPU capabilities before kernel execution
- Use appropriate buffer sizes for GPU memory
- Handle GPU errors gracefully with fallback to CPU
- Profile GPU vs CPU performance for your workload

## Migration from Other Libraries

### From NumPy/SciPy

```python
# NumPy
import numpy as np
result = np.dot(a, b)
normalized = result / np.linalg.norm(result)
```

```rust
// SciRS2-Core
use scirs2_core::simd_ops::SimdUnifiedOps;
let result = f64::simd_dot(&a.view(), &b.view());
let norm = f64::simd_norm(&result.view());
let normalized = f64::simd_mul(&result.view(), &Array1::from_elem(result.len(), 1.0 / norm).view());
```

### From ndarray

```rust
// ndarray
let result = &a + &b;
let sum = a.sum();

// SciRS2-Core (with SIMD acceleration)
let result = f64::simd_add(&a.view(), &b.view());
let sum = f64::simd_sum(&a.view());
```

## Next Steps

1. **Explore the API Reference**: Detailed documentation of all modules and functions
2. **Try the Examples**: Complete working examples for different use cases
3. **Performance Guide**: Advanced optimization techniques and benchmarking
4. **GPU Programming Guide**: Detailed GPU acceleration documentation
5. **Contributing Guide**: How to contribute to the SciRS2 ecosystem

For more detailed information, see the [API Reference](api_reference.md) and [Examples](examples.md).