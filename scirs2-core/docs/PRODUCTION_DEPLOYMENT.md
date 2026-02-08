# SciRS2 Core 1.0 Production Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Considerations](#security-considerations)
8. [Scalability and High Availability](#scalability-and-high-availability)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance and Updates](#maintenance-and-updates)
11. [Best Practices](#best-practices)

## Overview

SciRS2 Core 1.0 is a production-ready scientific computing library built in Rust, offering SciPy-compatible APIs with enhanced performance, safety, and concurrency. This guide covers enterprise-grade deployment considerations for production environments.

### Key Features for Production

- **API Stability**: Frozen 1.0 API surface with backward compatibility guarantees
- **Performance**: SIMD acceleration, multi-core parallelism, and GPU computing
- **Memory Efficiency**: Advanced memory management and out-of-core computation
- **Observability**: Comprehensive metrics, tracing, and audit logging
- **Security**: Built-in security features and vulnerability management
- **Cross-Platform**: Support for Linux, macOS, and Windows environments

## System Requirements

### Minimum Requirements

#### Hardware
- **CPU**: x86_64 or ARM64 architecture
- **Memory**: 4GB RAM (8GB recommended for large datasets)
- **Storage**: 1GB available disk space
- **Network**: Standard network connectivity for distributed features

#### Software
- **Rust**: 1.70.0 or later (MSRV - Minimum Supported Rust Version)
- **Operating System**: 
  - Linux (Ubuntu 20.04+, RHEL 8+, CentOS 8+)
  - macOS 11.0+
  - Windows 10 / Windows Server 2019+

### Recommended Production Requirements

#### Hardware
- **CPU**: Multi-core x86_64 with AVX2 support (Intel Haswell+ or AMD Zen+)
- **Memory**: 16GB+ RAM for optimal performance
- **Storage**: SSD with 10GB+ available space
- **GPU**: Optional NVIDIA GPU with CUDA 11.0+ or AMD GPU with ROCm 4.0+

#### Software Dependencies
- **BLAS/LAPACK**: OxiBLAS (pure Rust) - no system dependencies required
- **Compiler**: GCC 9+ or Clang 11+ for native compilation (optional, for FFI)

> **Note (v0.1.5+)**: SciRS2 now uses OxiBLAS for all BLAS/LAPACK operations. 
> No system libraries (OpenBLAS, MKL, Accelerate) are required.

## Installation

### Using Cargo (Recommended)

```toml
# Cargo.toml
[dependencies]
scirs2-core = { version = "1.0", features = ["production"] }

# Optional: Enable specific features for your use case
scirs2-core = { 
    version = "1.0", 
    features = [
        "production",    # All production features
        "parallel",      # Multi-core processing
        "simd",          # SIMD acceleration
        "gpu",           # GPU computing
        "memory_efficient", # Large dataset support
        "observability", # Monitoring and tracing
        "linalg",        # Linear algebra operations (OxiBLAS)
    ]
}
```

### Feature Flags for Production

#### Core Production Features
```toml
# Essential for production deployments
features = [
    "production",        # Enables all production features
    "observability",     # Metrics, tracing, audit logging
    "versioning",        # API compatibility management
    "data_validation",   # Comprehensive input validation
    "leak_detection",    # Memory leak detection
]
```

#### Performance Features
```toml
# For high-performance computing
features = [
    "simd",             # SIMD vector operations
    "parallel",         # Multi-core parallelism
    "gpu",              # GPU acceleration
    "memory_efficient", # Memory-mapped operations
    "linalg",           # Linear algebra (OxiBLAS - pure Rust)
]
```

#### GPU-Specific Features
```toml
# NVIDIA GPU support
features = ["cuda"]

# AMD GPU support  
features = ["rocm"]
```

### System Package Dependencies

> **Note**: BLAS/LAPACK system packages are no longer required. SciRS2 uses OxiBLAS (pure Rust).

#### Ubuntu/Debian
```bash
# Essential build dependencies
sudo apt update
sudo apt install build-essential pkg-config libssl-dev

# Optional: GPU support
sudo apt install nvidia-cuda-toolkit  # For NVIDIA
```

#### RHEL/CentOS/Fedora
```bash
# Essential build dependencies
sudo dnf install gcc pkg-config openssl-devel

# Optional: GPU support
sudo dnf install cuda-toolkit  # For NVIDIA (from NVIDIA repos)
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Optional: Install Homebrew dependencies
brew install pkg-config openssl
```

#### Windows
```powershell
# No special dependencies required for BLAS/LAPACK
# OxiBLAS (pure Rust) handles all linear algebra operations
```

## Configuration

### Environment Variables

#### Core Configuration
```bash
# Set the number of threads for parallel operations
export SCIRS2_NUM_THREADS=8

# Configure memory limits (in MB)
export SCIRS2_MEMORY_LIMIT=8192

# Enable/disable specific features
export SCIRS2_ENABLE_GPU=true
export SCIRS2_ENABLE_SIMD=true
export SCIRS2_ENABLE_VALIDATION=true
```

#### Performance Tuning
```bash
# Thread control (for compatibility with other libraries)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Memory allocation
export SCIRS2_CHUNK_SIZE=1048576      # 1MB chunks
export SCIRS2_PREFETCH_SIZE=4194304   # 4MB prefetch

# GPU configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1       # Use first two GPUs
```

#### Observability Configuration
```bash
# Enable comprehensive logging
export SCIRS2_LOG_LEVEL=info
export SCIRS2_ENABLE_TRACING=true
export SCIRS2_AUDIT_LOG=true

# Metrics collection
export SCIRS2_METRICS_ENDPOINT=http://prometheus:9090
export SCIRS2_METRICS_INTERVAL=30     # seconds

# Performance profiling
export SCIRS2_ENABLE_PROFILING=false  # Disable in production unless needed
```

### Configuration File

Create `/etc/scirs2/config.toml` or `~/.config/scirs2/config.toml`:

```toml
[runtime]
num_threads = 8
memory_limit_mb = 8192
enable_gpu = true
enable_simd = true

[performance]
chunk_size = 1048576
prefetch_size = 4194304
enable_adaptive_optimization = true

[observability]
log_level = "info"
enable_tracing = true
enable_audit_log = true
metrics_endpoint = "http://prometheus:9090"
metrics_interval = 30

[security]
enable_input_validation = true
enable_memory_protection = true
max_allocation_size = 1073741824  # 1GB

[features]
parallel = true
simd = true
gpu = false  # Override for specific environments
memory_efficient = true
```

### Runtime Configuration API

```rust
use scirs2_core::config::{Config, ConfigValue};

// Configure at runtime
let mut config = Config::default();
config.set("runtime.num_threads", ConfigValue::USize(8))?;
config.set("performance.enable_simd", ConfigValue::Bool(true))?;

// Apply configuration globally
scirs2_core::set_global_config(config)?;
```

## Performance Optimization

### SIMD Optimization

```rust
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities};

// Check platform capabilities
let capabilities = PlatformCapabilities::detect();
println!("SIMD support: {:?}", capabilities);

// Use SIMD operations for arrays
use ndarray::Array1;
let a = Array1::from(vec![1.0f32; 1000]);
let b = Array1::from(vec![2.0f32; 1000]);

// Automatic SIMD acceleration
let result = f32::simd_add(&a.view(), &b.view());
```

### Parallel Processing

```rust
use scirs2_core::parallel_ops::*;

// Configure parallel execution
set_num_threads(8);

// Parallel array operations
let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
let result: Vec<f64> = par_chunks(&data, 1000)
    .map(|chunk| chunk.iter().sum::<f64>())
    .collect();
```

### Memory Optimization

```rust
use scirs2_core::memory_efficient::*;
use scirs2_core::memory::{BufferPool, global_buffer_pool};

// Use memory-mapped arrays for large datasets
let mmap = create_mmap("large_data.bin", &[1_000_000])?;
let mut array = MemoryMappedArray::<f64>::from_mmap(mmap, &[1_000_000])?;

// Configure adaptive chunking
let chunking = AdaptiveChunkingBuilder::new()
    .with_chunk_size(1_048_576)  // 1MB chunks
    .with_memory_limit(8_589_934_592)  // 8GB limit
    .build()?;

// Process data in chunks
let result = chunk_wise_op(&array, |chunk| {
    chunk.mapv(|x| x * 2.0)
}, &chunking)?;
```

### GPU Acceleration

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend};

// Initialize GPU context
let gpu_context = GpuContext::new(GpuBackend::CUDA)?;

// GPU-accelerated operations
let gpu_array = gpu_context.allocate_buffer(data.len())?;
gpu_context.copy_to_device(&data, &gpu_array)?;

// Execute GPU kernel
let result = gpu_context.execute_kernel("vector_add", &[&gpu_array, &gpu_array])?;
```

## Monitoring and Observability

### Metrics Collection

```rust
use scirs2_core::metrics::{global_metrics_registry, Counter, Gauge, Timer};

// Built-in metrics
let registry = global_metrics_registry();
let operations_counter = registry.counter("scirs2_operations_total", "Total operations")?;
let memory_gauge = registry.gauge("scirs2_memory_usage_bytes", "Memory usage")?;
let compute_timer = registry.timer("scirs2_compute_duration", "Computation time")?;

// Record metrics
operations_counter.increment();
memory_gauge.set(1_073_741_824.0);  // 1GB
let _timer_guard = compute_timer.start();
```

### Tracing and Logging

```rust
use scirs2_core::observability::tracing::{trace_operation, TracingContext};

// Trace operations
let trace_ctx = TracingContext::new("scientific_computation");
trace_operation(&trace_ctx, "matrix_multiplication", || {
    // Your computation here
    Ok(result)
})?;
```

### Health Monitoring

```rust
use scirs2_core::metrics::{HealthMonitor, HealthCheck, HealthStatus};

// Built-in health checks
let health_monitor = HealthMonitor::new();
health_monitor.add_check(HealthCheck::new(
    "memory_usage",
    || {
        let usage = get_memory_usage()?;
        if usage < 0.9 { HealthStatus::Healthy } else { HealthStatus::Unhealthy }
    }
));

// Check system health
let status = health_monitor.check_health();
println!("System health: {:?}", status);
```

### Integration with Monitoring Systems

#### Prometheus Integration
```rust
use scirs2_core::observability::metrics::PrometheusExporter;

// Export metrics to Prometheus
let exporter = PrometheusExporter::new("0.0.0.0:9090")?;
exporter.start()?;
```

#### Jaeger Tracing
```rust
use scirs2_core::observability::tracing::JaegerExporter;

// Export traces to Jaeger
let jaeger = JaegerExporter::new("http://jaeger:14268/api/traces")?;
jaeger.start()?;
```

## Security Considerations

### Input Validation

```rust
use scirs2_core::validation::data::{ValidationSchema, Validator};

// Define validation schema
let schema = ValidationSchema::new()
    .field("input_data", DataType::FloatArray)
    .constraint("input_data", Constraint::FiniteValues)
    .constraint("input_data", Constraint::MaxSize(1_000_000));

// Validate input
let validator = Validator::new(schema);
validator.validate(&input_data)?;
```

### Memory Safety

```rust
use scirs2_core::memory::{LeakDetector, MemoryCheckpoint};

// Enable memory leak detection in development
#[cfg(debug_assertions)]
{
    let leak_detector = LeakDetector::new();
    let checkpoint = leak_detector.checkpoint();
    
    // Your code here
    
    leak_detector.check_leaks(&checkpoint)?;
}
```

### Secure Communication

```rust
use scirs2_core::observability::audit::{AuditLog, AuditEvent};

// Audit sensitive operations
let audit_log = AuditLog::new();
audit_log.log(AuditEvent::new(
    "data_access",
    "user123",
    "Accessed sensitive dataset",
))?;
```

### Best Practices

1. **Always validate inputs** in production environments
2. **Enable audit logging** for compliance requirements
3. **Use memory protection** features for critical applications
4. **Regularly update dependencies** for security patches
5. **Monitor resource usage** to prevent DoS conditions

## Scalability and High Availability

### Horizontal Scaling

```rust
use scirs2_core::distributed::{ClusterConfig, DistributedArray};

// Configure distributed computing
let cluster_config = ClusterConfig::new()
    .with_nodes(vec!["node1:8080", "node2:8080", "node3:8080"])
    .with_replication_factor(2);

// Distributed arrays
let distributed_data = DistributedArray::from_local_array(data, cluster_config)?;
let result = distributed_data.parallel_map(|chunk| process_chunk(chunk))?;
```

### Load Balancing

```rust
use scirs2_core::resource::{ResourceManager, LoadBalancer};

// Automatic load balancing
let load_balancer = LoadBalancer::new()
    .with_strategy(LoadBalancingStrategy::RoundRobin)
    .with_health_checks(true);

// Distribute work across available resources
let results = load_balancer.distribute_work(tasks)?;
```

### Fault Tolerance

```rust
use scirs2_core::error::recovery::{CircuitBreaker, RetryPolicy};

// Configure circuit breaker
let circuit_breaker = CircuitBreaker::new()
    .with_failure_threshold(5)
    .with_timeout(Duration::from_secs(30));

// Automatic retry with exponential backoff
let retry_policy = RetryPolicy::new()
    .with_max_attempts(3)
    .with_exponential_backoff(Duration::from_millis(100));

// Resilient operation execution
let result = circuit_breaker.execute_with_retry(|| {
    risky_computation()
}, retry_policy)?;
```

## Troubleshooting

### Common Issues

#### Performance Issues

**Problem**: Slower than expected performance
```bash
# Check CPU utilization
htop

# Check memory usage
free -h

# Check NUMA topology
numactl --hardware

# Profile the application
cargo build --release --features profiling
SCIRS2_ENABLE_PROFILING=true ./your_app
```

**Solution**: 
- Ensure SIMD is enabled and supported
- Verify optimal thread count (usually equal to CPU cores)
- Check memory bandwidth limitations
- Consider GPU acceleration for compute-intensive workloads

#### Memory Issues

**Problem**: Out of memory errors
```rust
// Enable memory monitoring
use scirs2_core::memory::metrics::{MemoryMetricsCollector, take_snapshot};

let collector = MemoryMetricsCollector::new();
let snapshot = take_snapshot();
println!("Memory usage: {:?}", snapshot);
```

**Solution**:
- Use memory-efficient operations for large datasets
- Enable adaptive chunking
- Increase system memory or reduce dataset size
- Use memory-mapped arrays for very large data

#### GPU Issues

**Problem**: GPU operations failing
```bash
# Check GPU status
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# Verify CUDA installation
nvcc --version

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Solution**:
- Verify GPU drivers are installed and up-to-date
- Check CUDA/ROCm toolkit installation
- Ensure sufficient GPU memory
- Verify CUDA_VISIBLE_DEVICES environment variable

### Debug Configuration

```toml
[debug]
enable_debug_logging = true
memory_tracking = true
performance_profiling = true
validate_all_operations = true

[logging]
level = "debug"
file = "/var/log/scirs2/debug.log"
max_size = "100MB"
max_files = 10
```

### Diagnostic Tools

```rust
use scirs2_core::diagnostic::{SystemInfo, PerformanceDiagnostic};

// System diagnostics
let system_info = SystemInfo::collect();
println!("System info: {:#?}", system_info);

// Performance diagnostics
let perf_diag = PerformanceDiagnostic::run_comprehensive_test();
println!("Performance report: {}", perf_diag.generate_report());
```

## Maintenance and Updates

### Updating SciRS2

#### Semantic Versioning
- **Patch updates (1.0.x)**: Bug fixes, safe to update automatically
- **Minor updates (1.x.0)**: New features, backward compatible
- **Major updates (x.0.0)**: Breaking changes, requires migration planning

#### Update Process
```bash
# Check current version
cargo metadata | grep scirs2-core

# Update to latest patch version
cargo update -p scirs2-core

# Update to specific version
cargo update -p scirs2-core@1.0.5
```

#### Compatibility Checking
```rust
use scirs2_core::versioning::{VersionManager, check_compatibility};

// Check API compatibility before upgrading
let version_manager = VersionManager::new();
let compatibility = version_manager.check_compatibility(
    &current_version,
    &target_version
)?;

println!("Compatibility: {:?}", compatibility);
```

### Backup and Recovery

#### Configuration Backup
```bash
# Backup configuration files
tar -czf scirs2-config-backup.tar.gz /etc/scirs2/ ~/.config/scirs2/

# Backup application data
rsync -av /path/to/data/ /backup/location/
```

#### Recovery Procedures
```bash
# Restore configuration
tar -xzf scirs2-config-backup.tar.gz -C /

# Verify system health after recovery
scirs2-health-check --comprehensive
```

### Performance Monitoring

```bash
# Monitor resource usage
watch -n 1 'ps aux | grep scirs2'

# Monitor memory allocation patterns
valgrind --tool=massif ./your_scirs2_app

# Profile CPU usage
perf record -g ./your_scirs2_app
perf report
```

## Best Practices

### Development Best Practices

1. **Use feature flags appropriately**
   ```toml
   # Development
   features = ["testing", "profiling", "debug"]
   
   # Production
   features = ["production", "parallel", "simd"]
   ```

2. **Implement proper error handling**
   ```rust
   use scirs2_core::{CoreResult, CoreError};
   
   fn robust_computation() -> CoreResult<f64> {
       let result = risky_operation()
           .map_err(|e| CoreError::ComputationError(e.into()))?;
       Ok(result)
   }
   ```

3. **Use validation for all inputs**
   ```rust
   use scirs2_core::validation::{check_finite, check_positive};
   
   fn safe_computation(data: &[f64], scale: f64) -> CoreResult<Vec<f64>> {
       check_finite(data, "input data")?;
       check_positive(scale, "scale factor")?;
       // Safe to proceed
   }
   ```

### Deployment Best Practices

1. **Use containerization for consistent environments**
   ```dockerfile
   FROM rust:1.70-slim
   
   # Install minimal system dependencies (OxiBLAS requires no external BLAS)
   RUN apt-get update && apt-get install -y \
       pkg-config \
       libssl-dev
   
   # Copy and build application
   COPY . /app
   WORKDIR /app
   RUN cargo build --release --features production
   
   # Runtime configuration
   ENV SCIRS2_NUM_THREADS=8
   ENV SCIRS2_ENABLE_SIMD=true
   
   CMD ["./target/release/your-app"]
   ```

2. **Implement health checks**
   ```rust
   use scirs2_core::metrics::{HealthMonitor, global_health_monitor};
   
   // HTTP health endpoint
   async fn health_check() -> impl Reply {
       let health = global_health_monitor().check_health();
       if health.is_healthy() {
           warp::reply::with_status("OK", StatusCode::OK)
       } else {
           warp::reply::with_status("UNHEALTHY", StatusCode::SERVICE_UNAVAILABLE)
       }
   }
   ```

3. **Configure proper logging**
   ```rust
   use scirs2_core::logging::{LogConfig, StructuredLogger};
   
   let log_config = LogConfig::new()
       .with_level("info")
       .with_file("/var/log/scirs2/app.log")
       .with_rotation(true)
       .with_structured_format(true);
   
   StructuredLogger::init(log_config)?;
   ```

### Security Best Practices

1. **Validate all external inputs**
2. **Use secure communication channels**
3. **Implement proper authentication and authorization**
4. **Enable audit logging for compliance**
5. **Regularly update dependencies**
6. **Use minimal privilege principles**

### Performance Best Practices

1. **Profile before optimizing**
2. **Use appropriate data structures**
3. **Leverage SIMD and parallel operations**
4. **Minimize memory allocations**
5. **Use memory-efficient algorithms for large datasets**
6. **Consider GPU acceleration for appropriate workloads**

## Support and Resources

### Documentation
- [API Documentation](https://docs.rs/scirs2-core/1.0)
- [User Guide](https://scirs2.org/guide)
- [Examples Repository](https://github.com/cool-japan/scirs/tree/main/examples)

### Community
- [GitHub Issues](https://github.com/cool-japan/scirs/issues)
- [Discussion Forum](https://github.com/cool-japan/scirs/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/scirs2)

### Professional Support
- Enterprise support available through official channels
- Training and consulting services
- Custom feature development

---

**Version**: SciRS2 Core 1.0  
**Last Updated**: 2025-06-29  
**Authors**: SciRS2 Development Team  
**License**: See LICENSE file for details