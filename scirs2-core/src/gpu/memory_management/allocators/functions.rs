//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::pool::{MemoryError, MemoryResult};

use super::types::{BuddyAllocator, HybridAllocator, SlabAllocator};

/// Benchmark module for comparing allocator performance
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    /// Benchmark result for a single allocator
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub allocator_name: String,
        pub total_time_ns: u128,
        pub avg_allocation_ns: f64,
        pub avg_deallocation_ns: f64,
        pub final_fragmentation: f64,
        pub peak_memory_usage: usize,
        pub operations_per_second: f64,
    }
    /// Benchmark configuration
    #[derive(Debug, Clone)]
    pub struct BenchmarkConfig {
        pub num_iterations: usize,
        pub allocation_sizes: Vec<usize>,
        pub pattern: AllocationPattern,
    }
    /// Allocation pattern for benchmarking
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AllocationPattern {
        Sequential,
        Interleaved,
        RandomSizes,
        FragmentationTest,
    }
    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                num_iterations: 10000,
                allocation_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
                pattern: AllocationPattern::Sequential,
            }
        }
    }
    /// Benchmark BuddyAllocator
    pub fn benchmark_buddy_allocator(config: &BenchmarkConfig) -> MemoryResult<BenchmarkResult> {
        let allocator = BuddyAllocator::new(256 * 1024 * 1024, 64)?;
        let start = Instant::now();
        let mut allocations = Vec::new();
        let mut total_alloc_time = 0u128;
        let mut total_dealloc_time = 0u128;
        match config.pattern {
            AllocationPattern::Sequential => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    let result = allocator.allocate(size);
                    total_alloc_time += alloc_start.elapsed().as_nanos();
                    if let Ok(offset) = result {
                        allocations.push(offset);
                    }
                }
                for offset in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            AllocationPattern::Interleaved => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push(offset);
                        if i % 2 == 1 && !allocations.is_empty() {
                            let dealloc_offset = allocations.remove(0);
                            let dealloc_start = Instant::now();
                            let _ = allocator.deallocate(dealloc_offset);
                            total_dealloc_time += dealloc_start.elapsed().as_nanos();
                        }
                    }
                }
                for offset in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            _ => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push(offset);
                    }
                }
                for offset in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
        }
        let total_time = start.elapsed().as_nanos();
        let stats = allocator.statistics();
        Ok(BenchmarkResult {
            allocator_name: "BuddyAllocator".to_string(),
            total_time_ns: total_time,
            avg_allocation_ns: total_alloc_time as f64 / config.num_iterations as f64,
            avg_deallocation_ns: total_dealloc_time as f64 / config.num_iterations as f64,
            final_fragmentation: stats.fragmentation_ratio,
            peak_memory_usage: stats.current_allocated_bytes,
            operations_per_second: (config.num_iterations as f64 * 2.0)
                / (total_time as f64 / 1_000_000_000.0),
        })
    }
    /// Benchmark SlabAllocator
    pub fn benchmark_slab_allocator(config: &BenchmarkConfig) -> MemoryResult<BenchmarkResult> {
        let allocator = SlabAllocator::new();
        let start = Instant::now();
        let mut allocations = Vec::new();
        let mut total_alloc_time = 0u128;
        let mut total_dealloc_time = 0u128;
        match config.pattern {
            AllocationPattern::Sequential => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    let result = allocator.allocate(size);
                    total_alloc_time += alloc_start.elapsed().as_nanos();
                    if let Ok(offset) = result {
                        allocations.push((offset, size));
                    }
                }
                for (offset, _) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            AllocationPattern::Interleaved => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push((offset, size));
                        if i % 2 == 1 && !allocations.is_empty() {
                            let (dealloc_offset, _) = allocations.remove(0);
                            let dealloc_start = Instant::now();
                            let _ = allocator.deallocate(dealloc_offset);
                            total_dealloc_time += dealloc_start.elapsed().as_nanos();
                        }
                    }
                }
                for (offset, _) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            _ => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push((offset, size));
                    }
                }
                for (offset, _) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
        }
        let total_time = start.elapsed().as_nanos();
        let stats = allocator.statistics();
        Ok(BenchmarkResult {
            allocator_name: "SlabAllocator".to_string(),
            total_time_ns: total_time,
            avg_allocation_ns: total_alloc_time as f64 / config.num_iterations as f64,
            avg_deallocation_ns: total_dealloc_time as f64 / config.num_iterations as f64,
            final_fragmentation: stats.fragmentation_ratio,
            peak_memory_usage: stats.current_allocated_bytes,
            operations_per_second: (config.num_iterations as f64 * 2.0)
                / (total_time as f64 / 1_000_000_000.0),
        })
    }
    /// Benchmark HybridAllocator
    pub fn benchmark_hybrid_allocator(config: &BenchmarkConfig) -> MemoryResult<BenchmarkResult> {
        let allocator = HybridAllocator::new()?;
        let start = Instant::now();
        let mut allocations = Vec::new();
        let mut total_alloc_time = 0u128;
        let mut total_dealloc_time = 0u128;
        match config.pattern {
            AllocationPattern::Sequential => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    let result = allocator.allocate(size);
                    total_alloc_time += alloc_start.elapsed().as_nanos();
                    if let Ok(offset) = result {
                        allocations.push((offset, size));
                    }
                }
                for (offset, size) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset, size);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            AllocationPattern::Interleaved => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push((offset, size));
                        if i % 2 == 1 && !allocations.is_empty() {
                            let (dealloc_offset, dealloc_size) = allocations.remove(0);
                            let dealloc_start = Instant::now();
                            let _ = allocator.deallocate(dealloc_offset, dealloc_size);
                            total_dealloc_time += dealloc_start.elapsed().as_nanos();
                        }
                    }
                }
                for (offset, size) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset, size);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
            _ => {
                for i in 0..config.num_iterations {
                    let size = config.allocation_sizes[i % config.allocation_sizes.len()];
                    let alloc_start = Instant::now();
                    if let Ok(offset) = allocator.allocate(size) {
                        total_alloc_time += alloc_start.elapsed().as_nanos();
                        allocations.push((offset, size));
                    }
                }
                for (offset, size) in allocations {
                    let dealloc_start = Instant::now();
                    let _ = allocator.deallocate(offset, size);
                    total_dealloc_time += dealloc_start.elapsed().as_nanos();
                }
            }
        }
        let total_time = start.elapsed().as_nanos();
        let stats = allocator.statistics();
        let total_mem = stats.slab_statistics.total_memory + stats.buddy_statistics.total_memory;
        let total_used = stats.slab_statistics.current_allocated_bytes
            + stats.buddy_statistics.current_allocated_bytes;
        let fragmentation = if total_mem > 0 {
            1.0 - (total_used as f64 / total_mem as f64)
        } else {
            0.0
        };
        Ok(BenchmarkResult {
            allocator_name: "HybridAllocator".to_string(),
            total_time_ns: total_time,
            avg_allocation_ns: total_alloc_time as f64 / config.num_iterations as f64,
            avg_deallocation_ns: total_dealloc_time as f64 / config.num_iterations as f64,
            final_fragmentation: fragmentation,
            peak_memory_usage: total_used,
            operations_per_second: (config.num_iterations as f64 * 2.0)
                / (total_time as f64 / 1_000_000_000.0),
        })
    }
    /// Run comprehensive benchmark comparing all allocators
    pub fn run_comprehensive_benchmark() -> MemoryResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        let config = BenchmarkConfig {
            num_iterations: 5000,
            allocation_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
            pattern: AllocationPattern::Sequential,
        };
        results.push(benchmark_buddy_allocator(&config)?);
        results.push(benchmark_slab_allocator(&config)?);
        results.push(benchmark_hybrid_allocator(&config)?);
        let config = BenchmarkConfig {
            num_iterations: 5000,
            allocation_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
            pattern: AllocationPattern::Interleaved,
        };
        results.push(benchmark_buddy_allocator(&config)?);
        results.push(benchmark_slab_allocator(&config)?);
        results.push(benchmark_hybrid_allocator(&config)?);
        Ok(results)
    }
    /// Print benchmark results in a formatted table
    pub fn print_benchmark_results(results: &[BenchmarkResult]) {
        println!("\n{:=<100}", "");
        println!("GPU Memory Allocator Benchmark Results");
        println!("{:=<100}", "");
        println!(
            "{:<20} {:>15} {:>15} {:>15} {:>15} {:>18}",
            "Allocator",
            "Total Time (ms)",
            "Alloc (ns)",
            "Dealloc (ns)",
            "Fragmentation",
            "Ops/sec"
        );
        println!("{:-<100}", "");
        for result in results {
            println!(
                "{:<20} {:>15.2} {:>15.2} {:>15.2} {:>14.2}% {:>18.0}",
                result.allocator_name,
                result.total_time_ns as f64 / 1_000_000.0,
                result.avg_allocation_ns,
                result.avg_deallocation_ns,
                result.final_fragmentation * 100.0,
                result.operations_per_second
            );
        }
        println!("{:=<100}\n", "");
    }
}
