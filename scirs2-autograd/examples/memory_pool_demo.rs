// Memory pool demonstration for scirs2-autograd.
//
// This example showcases the TensorPool memory pool:
// 1. TensorPool - general-purpose tensor pooling by shape
// 2. Per-bucket limits (LRU-style eviction)
// 3. Gradient accumulation simulation
// 4. Performance comparison vs direct allocation

use scirs2_autograd::memory_pool::{global_pool, PooledArray, TensorPool};
use scirs2_autograd::ndarray_ext::NdArray;
use scirs2_core::ndarray;
use std::time::Instant;

fn main() {
    println!("=== Memory Pool Demonstration ===\n");

    demo_tensor_pool();
    println!();
    demo_limited_pool();
    println!();
    demo_gradient_simulation();
    println!();
    demo_performance_comparison();
}

fn demo_tensor_pool() {
    println!("1. TensorPool - General-purpose tensor pooling");
    println!("------------------------------------------------");

    let pool = TensorPool::new();

    // Acquire and release buffers.
    println!("Acquiring 5 buffers with shape [64, 64]...");
    let mut buffers = Vec::new();
    for _ in 0..5 {
        let buf: PooledArray<f64> = pool.acquire(&[64, 64]);
        buffers.push(buf);
    }

    println!("Stats after acquisition: {}", pool.stats());

    // Release all buffers.
    drop(buffers);
    println!("Stats after release: {}", pool.stats());

    // Re-acquire (should reuse).
    println!("\nRe-acquiring 3 buffers...");
    let _buf1: PooledArray<f64> = pool.acquire(&[64, 64]);
    let _buf2: PooledArray<f64> = pool.acquire(&[64, 64]);
    let _buf3: PooledArray<f64> = pool.acquire(&[64, 64]);

    let stats = pool.stats();
    println!("Stats after re-acquisition: {}", stats);
    if stats.n_acquired > 0 {
        println!(
            "Reuse rate: {:.1}%",
            100.0 * stats.n_reused as f64 / stats.n_acquired as f64
        );
    }
}

fn demo_limited_pool() {
    println!("2. Limited TensorPool - Per-bucket size cap");
    println!("--------------------------------------------");

    // Create a pool that keeps at most 3 buffers per shape bucket.
    let pool = TensorPool::with_max_per_bucket(3);

    println!("Acquiring buffers of various sizes...");
    let buf1: PooledArray<f64> = pool.acquire(&[1000]);
    let buf2: PooledArray<f64> = pool.acquire(&[2000]);
    let buf3: PooledArray<f64> = pool.acquire(&[1000]);

    println!("Stats after acquisition: {}", pool.stats());

    drop(buf1);
    drop(buf2);
    drop(buf3);

    println!("Stats after release: {}", pool.stats());

    // With max 3 per bucket, overflow cycles should still report pooled buffers.
    println!("\nTesting per-bucket cap (max 3)...");
    for i in 0..5 {
        let buf: PooledArray<f64> = pool.acquire(&[5000]);
        drop(buf);
        let s = pool.stats();
        println!(
            "  After {} releases: {} pooled buffers across {} buckets",
            i + 1,
            s.n_pooled_buffers,
            s.n_buckets
        );
    }
}

fn demo_gradient_simulation() {
    println!("3. Gradient Accumulation Simulation");
    println!("------------------------------------");

    // Simulate a backward pass that acquires per-layer gradient buffers,
    // uses them, and returns them for reuse in the next pass.
    let pool = TensorPool::new();

    let layer_shapes: Vec<Vec<usize>> = vec![
        vec![32, 64],
        vec![64, 128],
        vec![128, 256],
        vec![256, 128],
        vec![128, 64],
    ];

    println!(
        "Simulating 2 backward passes over {} layers...",
        layer_shapes.len()
    );

    // First backward pass.
    let mut grads: Vec<PooledArray<f64>> = layer_shapes
        .iter()
        .map(|shape| pool.acquire(shape))
        .collect();

    let s1 = pool.stats();
    println!(
        "After first pass:  acquired={}, allocated={}, reused={}",
        s1.n_acquired, s1.n_allocated, s1.n_reused
    );

    // Release gradients (simulate optimizer step returning buffers).
    grads.clear();

    // Second backward pass (should reuse all buffers).
    let _grads2: Vec<PooledArray<f64>> = layer_shapes
        .iter()
        .map(|shape| pool.acquire(shape))
        .collect();

    let s2 = pool.stats();
    println!(
        "After second pass: acquired={}, allocated={}, reused={}",
        s2.n_acquired, s2.n_allocated, s2.n_reused
    );
    if s2.n_acquired > 0 {
        println!(
            "Overall reuse rate: {:.1}%",
            100.0 * s2.n_reused as f64 / s2.n_acquired as f64
        );
    }
}

fn demo_performance_comparison() {
    println!("4. Performance Comparison");
    println!("-------------------------");

    let n_iterations = 10_000;
    let shape: &[usize] = &[128, 128];

    // Direct allocation (baseline).
    let start = Instant::now();
    for _ in 0..n_iterations {
        let _arr: NdArray<f64> = ndarray::Array::zeros(ndarray::IxDyn(shape));
    }
    let direct_duration = start.elapsed();

    // TensorPool via global singleton (warm cache).
    let gpool = global_pool();
    // Pre-warm: acquire + drop 10 times so the bucket is populated.
    for _ in 0..10 {
        let buf: PooledArray<f64> = gpool.acquire(shape);
        drop(buf);
    }
    let start = Instant::now();
    for _ in 0..n_iterations {
        let buf: PooledArray<f64> = gpool.acquire(shape);
        drop(buf);
    }
    let global_pool_duration = start.elapsed();

    // Dedicated TensorPool (warm cache).
    let pool = TensorPool::new();
    for _ in 0..10 {
        let buf: PooledArray<f64> = pool.acquire(shape);
        drop(buf);
    }
    let start = Instant::now();
    for _ in 0..n_iterations {
        let buf: PooledArray<f64> = pool.acquire(shape);
        drop(buf);
    }
    let pool_duration = start.elapsed();

    // TensorPool with per-bucket cap (bounded memory usage).
    let bounded_pool = TensorPool::with_max_per_bucket(4);
    for _ in 0..4 {
        let buf: PooledArray<f64> = bounded_pool.acquire(shape);
        drop(buf);
    }
    let start = Instant::now();
    for _ in 0..n_iterations {
        let buf: PooledArray<f64> = bounded_pool.acquire(shape);
        drop(buf);
    }
    let bounded_duration = start.elapsed();

    println!("\n{} iterations with shape {:?}:", n_iterations, shape);
    println!(
        "  Direct allocation:         {:>8.2}ms (baseline)",
        direct_duration.as_secs_f64() * 1000.0
    );
    let gspeedup = direct_duration.as_secs_f64() / global_pool_duration.as_secs_f64().max(1e-9);
    println!(
        "  global_pool (singleton):   {:>8.2}ms ({:.2}x speedup)",
        global_pool_duration.as_secs_f64() * 1000.0,
        gspeedup
    );
    let speedup = direct_duration.as_secs_f64() / pool_duration.as_secs_f64().max(1e-9);
    println!(
        "  TensorPool (dedicated):    {:>8.2}ms ({:.2}x speedup)",
        pool_duration.as_secs_f64() * 1000.0,
        speedup
    );
    let bspeedup = direct_duration.as_secs_f64() / bounded_duration.as_secs_f64().max(1e-9);
    println!(
        "  TensorPool (bounded cap):  {:>8.2}ms ({:.2}x speedup)",
        bounded_duration.as_secs_f64() * 1000.0,
        bspeedup
    );

    println!("\nFinal statistics:");
    println!("  Dedicated pool: {}", pool.stats());
    println!("  Bounded pool:   {}", bounded_pool.stats());
    println!("  Global pool:    {}", gpool.stats());
}
