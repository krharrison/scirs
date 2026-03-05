//! SciRS2 v0.2.0 Memory Profiling Benchmarks
//!
//! This benchmark suite profiles memory usage patterns and efficiency:
//! - Heap allocation tracking
//! - Memory bandwidth utilization
//! - Cache efficiency measurements
//! - Out-of-core performance validation
//! - Memory pool effectiveness
//! - Zero-copy operation verification
//!
//! Performance Targets:
//! - Memory allocations: <5% overhead vs pre-allocated
//! - Cache efficiency: >80% L1 hit rate for small matrices
//! - Memory bandwidth: >50% of theoretical peak
//! - Out-of-core: <20% performance degradation

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use sysinfo::System;

// =============================================================================
// Memory Tracking Allocator
// =============================================================================

struct TrackingAllocator {
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl TrackingAllocator {
    const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.deallocated.store(0, Ordering::SeqCst);
        self.allocation_count.store(0, Ordering::SeqCst);
    }

    fn bytes_allocated(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    fn bytes_deallocated(&self) -> usize {
        self.deallocated.load(Ordering::SeqCst)
    }

    fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::SeqCst)
    }

    fn current_usage(&self) -> usize {
        self.bytes_allocated()
            .saturating_sub(self.bytes_deallocated())
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = SystemAllocator.alloc(layout);
        if !ptr.is_null() {
            self.allocated.fetch_add(layout.size(), Ordering::SeqCst);
            self.allocation_count.fetch_add(1, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        SystemAllocator.dealloc(ptr, layout);
        self.deallocated.fetch_add(layout.size(), Ordering::SeqCst);
    }
}

// Note: We cannot override the global allocator in a benchmark,
// so we'll use system monitoring instead

// =============================================================================
// Memory Usage Monitoring
// =============================================================================

struct MemoryMonitor {
    system: System,
    initial_memory: u64,
}

impl MemoryMonitor {
    fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_memory();
        let initial_memory = system.used_memory();

        Self {
            system,
            initial_memory,
        }
    }

    fn refresh(&mut self) {
        self.system.refresh_memory();
    }

    fn current_usage(&self) -> u64 {
        self.system.used_memory()
    }

    fn delta_usage(&self) -> i64 {
        (self.current_usage() as i64) - (self.initial_memory as i64)
    }

    fn peak_usage(&mut self) -> u64 {
        self.refresh();
        self.current_usage()
    }
}

// =============================================================================
// Memory Allocation Patterns
// =============================================================================

/// Benchmark allocation overhead
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation_overhead");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [1024, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f64>()) as u64,
        ));

        // Vec allocation
        group.bench_with_input(BenchmarkId::new("vec_alloc", size), &size, |b, &s| {
            b.iter(|| {
                let v = vec![0.0f64; s];
                black_box(v)
            })
        });

        // Pre-allocated with capacity
        group.bench_with_input(
            BenchmarkId::new("vec_with_capacity", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut v = Vec::with_capacity(s);
                    v.resize(s, 0.0f64);
                    black_box(v)
                })
            },
        );

        // Uninitialized allocation (unsafe but fast)
        group.bench_with_input(BenchmarkId::new("vec_uninit", size), &size, |b, &s| {
            b.iter(|| {
                let v = Vec::<f64>::with_capacity(s);
                black_box(v)
            })
        });
    }

    group.finish();
}

/// Benchmark memory reuse patterns
fn bench_memory_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/reuse_patterns");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let size = 100_000;
    let iterations = 100;

    group.throughput(Throughput::Elements((size * iterations) as u64));

    // New allocation each iteration
    group.bench_function("new_each_time", |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let v = vec![0.0f64; size];
                black_box(v);
            }
        })
    });

    // Reuse same buffer
    group.bench_function("reuse_buffer", |b| {
        b.iter(|| {
            let mut v = vec![0.0f64; size];
            for _ in 0..iterations {
                v.fill(0.0);
                black_box(&v);
            }
        })
    });

    // Reuse with clear
    group.bench_function("reuse_with_clear", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(size);
            for _ in 0..iterations {
                v.clear();
                v.resize(size, 0.0f64);
                black_box(&v);
            }
        })
    });

    group.finish();
}

// =============================================================================
// Memory Bandwidth Benchmarks
// =============================================================================

/// Benchmark memory bandwidth utilization
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/bandwidth");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [1024, 10_000, 100_000, 1_000_000, 10_000_000];

    for &size in &sizes {
        let src = vec![1.0f64; size];
        let mut dst = vec![0.0f64; size];

        let bytes = (size * std::mem::size_of::<f64>()) as u64;
        group.throughput(Throughput::Bytes(bytes));

        // Sequential copy
        group.bench_with_input(BenchmarkId::new("copy", size), &size, |b, _| {
            b.iter(|| {
                dst.copy_from_slice(&src);
                black_box(&dst);
            })
        });

        // Sequential read
        group.bench_with_input(BenchmarkId::new("read", size), &size, |b, _| {
            b.iter(|| {
                let sum: f64 = src.iter().sum();
                black_box(sum)
            })
        });

        // Sequential write
        group.bench_with_input(BenchmarkId::new("write", size), &size, |b, _| {
            b.iter(|| {
                dst.fill(1.0);
                black_box(&dst);
            })
        });

        // Read-modify-write
        group.bench_with_input(
            BenchmarkId::new("read_modify_write", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for i in 0..size {
                        dst[i] = src[i] * 2.0;
                    }
                    black_box(&dst);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark cache efficiency
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/cache_efficiency");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Sizes targeting different cache levels
    let configs = [
        (1024, "L1"),      // 8 KB - fits in L1
        (8192, "L2"),      // 64 KB - fits in L2
        (131072, "L3"),    // 1 MB - fits in L3
        (2097152, "DRAM"), // 16 MB - exceeds cache
    ];

    for &(size, label) in &configs {
        let data = vec![1.0f64; size];

        group.throughput(Throughput::Bytes((size * 8) as u64));

        // Sequential access (cache-friendly)
        group.bench_with_input(BenchmarkId::new("sequential", label), label, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f64;
                for &val in &data {
                    sum += val;
                }
                black_box(sum)
            })
        });

        // Strided access (cache-unfriendly)
        let stride = 16;
        group.bench_with_input(BenchmarkId::new("strided", label), label, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f64;
                let mut i = 0;
                while i < size {
                    sum += data[i];
                    i += stride;
                }
                black_box(sum)
            })
        });

        // Random access (cache-hostile)
        let indices: Vec<usize> = (0..size).map(|i| (i * 7919) % size).collect();
        group.bench_with_input(BenchmarkId::new("random", label), label, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f64;
                for &idx in &indices {
                    sum += data[idx];
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Chunked Processing Benchmarks
// =============================================================================

/// Benchmark chunked processing overhead
fn bench_chunked_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/chunked_processing");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let data_size = 1_000_000;
    let data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();

    group.throughput(Throughput::Elements(data_size as u64));

    // No chunking
    group.bench_function("no_chunks", |b| {
        b.iter(|| {
            let sum: f64 = data.iter().sum();
            black_box(sum)
        })
    });

    // Fixed chunk sizes
    for &chunk_size in &[1024, 8192, 65536] {
        let label = format!("chunk_{}", chunk_size);
        group.bench_with_input(BenchmarkId::new("fixed", &label), &label, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f64;
                for chunk in data.chunks(chunk_size) {
                    sum += chunk.iter().sum::<f64>();
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Out-of-Core Processing Benchmarks
// =============================================================================

/// Benchmark out-of-core operations
fn bench_out_of_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/out_of_core");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Simulate large dataset that doesn't fit in memory
    // Process in chunks
    let total_size = 10_000_000;
    let chunk_sizes = [10_000, 100_000, 1_000_000];

    for &chunk_size in &chunk_sizes {
        let label = format!("chunk_{}", chunk_size);
        group.throughput(Throughput::Elements(total_size as u64));

        group.bench_with_input(
            BenchmarkId::new("process_chunks", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let mut global_sum = 0.0f64;
                    let n_chunks = (total_size + chunk_size - 1) / chunk_size;

                    for chunk_idx in 0..n_chunks {
                        let start = chunk_idx * chunk_size;
                        let end = (start + chunk_size).min(total_size);
                        let current_chunk_size = end - start;

                        // Simulate loading chunk from disk
                        let chunk: Vec<f64> = (0..current_chunk_size)
                            .map(|i| (start + i) as f64)
                            .collect();

                        // Process chunk
                        let chunk_sum: f64 = chunk.iter().sum();
                        global_sum += chunk_sum;
                    }

                    black_box(global_sum)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Memory Pool Benchmarks
// =============================================================================

/// Simple memory pool for benchmarking
struct SimplePool {
    buffers: Vec<Vec<f64>>,
    size: usize,
}

impl SimplePool {
    fn new(capacity: usize, buffer_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffers.push(Vec::with_capacity(buffer_size));
        }
        Self {
            buffers,
            size: buffer_size,
        }
    }

    fn acquire(&mut self) -> Option<Vec<f64>> {
        self.buffers.pop()
    }

    fn release(&mut self, mut buffer: Vec<f64>) {
        buffer.clear();
        if buffer.capacity() >= self.size {
            self.buffers.push(buffer);
        }
    }
}

/// Benchmark memory pool effectiveness
fn bench_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/pool");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let buffer_size = 10_000;
    let iterations = 1000;

    group.throughput(Throughput::Elements((buffer_size * iterations) as u64));

    // Without pool (allocate each time)
    group.bench_function("no_pool", |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let v = vec![0.0f64; buffer_size];
                black_box(v);
            }
        })
    });

    // With pool
    group.bench_function("with_pool", |b| {
        b.iter(|| {
            let mut pool = SimplePool::new(10, buffer_size);

            for _ in 0..iterations {
                let mut buffer = pool
                    .acquire()
                    .unwrap_or_else(|| Vec::with_capacity(buffer_size));
                buffer.resize(buffer_size, 0.0);
                black_box(&buffer);
                pool.release(buffer);
            }
        })
    });

    group.finish();
}

// =============================================================================
// Zero-Copy Operations
// =============================================================================

/// Benchmark zero-copy vs copying operations
fn bench_zero_copy(c: &mut Criterion) {
    use scirs2_core::ndarray::Array2;

    let mut group = c.benchmark_group("memory/zero_copy");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 500, 1000];

    for &size in &sizes {
        let matrix = Array2::<f64>::zeros((size, size));

        group.throughput(Throughput::Elements((size * size) as u64));

        // Copy (clone)
        group.bench_with_input(BenchmarkId::new("copy", size), &size, |b, _| {
            b.iter(|| {
                let copied = matrix.clone();
                black_box(copied)
            })
        });

        // View (zero-copy)
        group.bench_with_input(BenchmarkId::new("view", size), &size, |b, _| {
            b.iter(|| {
                let view = matrix.view();
                black_box(view)
            })
        });

        // Slice (zero-copy)
        let slice_size = size / 2;
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let slice = matrix.slice(ndarray::s![0..slice_size, 0..slice_size]);
                black_box(slice)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    allocation_benchmarks,
    bench_allocation_overhead,
    bench_memory_reuse,
);

criterion_group!(
    bandwidth_benchmarks,
    bench_memory_bandwidth,
    bench_cache_efficiency,
);

criterion_group!(
    chunking_benchmarks,
    bench_chunked_processing,
    bench_out_of_core,
);

criterion_group!(pool_benchmarks, bench_memory_pool,);

criterion_group!(zero_copy_benchmarks, bench_zero_copy,);

criterion_main!(
    allocation_benchmarks,
    bandwidth_benchmarks,
    chunking_benchmarks,
    pool_benchmarks,
    zero_copy_benchmarks,
);
