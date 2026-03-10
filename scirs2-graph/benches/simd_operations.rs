//! SIMD-accelerated graph operations benchmarks
//!
//! Compares performance of SIMD vs non-SIMD implementations for:
//! - PageRank power iteration
//! - Spectral Laplacian construction
//! - Matrix-vector products
//! - Graph traversal operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_graph::error::Result;
use scirs2_graph::spectral::LaplacianType;
use scirs2_graph::{cycle_graph, erdos_renyi_graph, star_graph, DiGraph, Graph, Node};
use std::hint::black_box;

#[cfg(feature = "simd")]
use scirs2_graph::simd_ops::{SimdAdjacency, SimdPageRank, SimdSpectral};

// Helper to create test graphs of various sizes
fn create_test_graphs(size: usize) -> Vec<Graph<usize, f64>> {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);
    let n = size.max(3);
    vec![
        cycle_graph(n).expect("failed to create cycle graph"),
        star_graph(n).expect("failed to create star graph"),
        erdos_renyi_graph(n, 0.1, &mut rng).expect("failed to create erdos_renyi graph"),
    ]
}

// Helper to create directed test graphs
fn create_test_digraphs(size: usize) -> Vec<DiGraph<usize, f64>> {
    let mut g = DiGraph::<usize, f64>::new();
    for i in 0..size {
        g.add_node(i);
    }
    for i in 0..size.saturating_sub(1) {
        let _ = g.add_edge(i, i + 1, 1.0);
    }
    vec![g]
}

// Benchmark PageRank computation
fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let graphs = create_test_digraphs(*size);

        #[cfg(feature = "simd")]
        {
            group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
                let graph = &graphs[0];
                let n = graph.node_count();

                // Build transition matrix
                let mut transition = Array2::zeros((n, n));
                let out_degrees = graph.out_degree_vector();

                for (i, node_idx) in graph.inner().node_indices().enumerate() {
                    let node_out_degree = out_degrees[i] as f64;
                    if node_out_degree > 0.0 {
                        for neighbor_idx in graph
                            .inner()
                            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                        {
                            let j = neighbor_idx.index();
                            transition[[i, j]] = 1.0 / node_out_degree;
                        }
                    }
                }

                b.iter(|| {
                    black_box(
                        SimdPageRank::compute_pagerank(&transition, 0.85, 1e-6, 100)
                            .expect("pagerank failed"),
                    )
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, _| {
            let graph = &graphs[0];
            b.iter(|| black_box(scirs2_graph::pagerank(graph, 0.85, 1e-6, 100)));
        });
    }

    group.finish();
}

// Benchmark Laplacian matrix construction
fn bench_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian");

    for size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements((size * size) as u64));

        let graphs = create_test_graphs(*size);
        let graph = &graphs[0];

        // Prepare data for SIMD version
        let adj_mat = graph.adjacency_matrix();
        let n = graph.node_count();
        let mut adj_f64 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                adj_f64[[i, j]] = adj_mat[[i, j]];
            }
        }
        let degrees = graph.degree_vector();
        let degrees_f64 = Array1::from_shape_fn(n, |i| degrees[i] as f64);

        #[cfg(feature = "simd")]
        {
            group.bench_with_input(BenchmarkId::new("simd_standard", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        SimdSpectral::standard_laplacian(&adj_f64, &degrees_f64)
                            .expect("standard_laplacian failed"),
                    )
                });
            });

            group.bench_with_input(BenchmarkId::new("simd_normalized", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        SimdSpectral::normalized_laplacian(&adj_f64, &degrees_f64)
                            .expect("normalized_laplacian failed"),
                    )
                });
            });

            group.bench_with_input(BenchmarkId::new("simd_random_walk", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        SimdSpectral::random_walk_laplacian(&adj_f64, &degrees_f64)
                            .expect("random_walk_laplacian failed"),
                    )
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    scirs2_graph::laplacian(graph, LaplacianType::Standard)
                        .expect("laplacian failed"),
                )
            });
        });
    }

    group.finish();
}

// Benchmark matrix-vector products
#[cfg(feature = "simd")]
fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");

    for size in [10, 50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements((size * size) as u64));

        let matrix = Array2::from_shape_fn((*size, *size), |(i, j)| {
            if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });
        let vector = Array1::from_elem(*size, 1.0);

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    SimdAdjacency::dense_matvec(&matrix, &vector).expect("dense_matvec failed"),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, _| {
            b.iter(|| {
                let result = matrix.dot(&vector);
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark power iteration for eigenvalues
#[cfg(feature = "simd")]
fn bench_power_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_iteration");

    for size in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Create a symmetric matrix (graph Laplacian-like)
        let matrix = Array2::from_shape_fn((*size, *size), |(i, j)| {
            if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });
        let initial = Array1::from_elem(*size, 1.0 / (*size as f64).sqrt());

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    SimdSpectral::power_iteration(&matrix, &initial, 50, 1e-10)
                        .expect("power_iteration failed"),
                )
            });
        });
    }

    group.finish();
}

// Benchmark sparse CSR matrix-vector product
#[cfg(feature = "simd")]
fn bench_sparse_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matvec");

    for size in [100, 500, 1000, 5000].iter() {
        // Create a sparse tridiagonal matrix in CSR format
        let nnz = size * 3 - 2; // Tridiagonal has at most 3n-2 non-zeros
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for i in 0..*size {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(-1.0);
            }
            col_idx.push(i);
            values.push(2.0);
            if i < size - 1 {
                col_idx.push(i + 1);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len());
        }

        let x = vec![1.0; *size];

        group.throughput(Throughput::Elements(nnz as u64));

        group.bench_with_input(BenchmarkId::new("simd_csr", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    SimdAdjacency::sparse_csr_matvec(&row_ptr, &col_idx, &values, &x, *size)
                        .expect("sparse_csr_matvec failed"),
                )
            });
        });
    }

    group.finish();
}

// Benchmark SIMD operations for betweenness centrality accumulation
#[cfg(feature = "simd")]
fn bench_betweenness_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("betweenness");

    for size in [50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let mut delta = vec![0.0; *size];
        let sigma = vec![1.0; *size];
        let predecessors: Vec<Vec<usize>> = (0..*size)
            .map(|i| if i > 0 { vec![i - 1] } else { vec![] })
            .collect();
        let order: Vec<usize> = (0..*size).collect();

        group.bench_with_input(BenchmarkId::new("simd_accumulate", size), size, |b, _| {
            b.iter(|| {
                delta.fill(0.0);
                scirs2_graph::simd_ops::SimdBetweenness::accumulate_dependencies(
                    black_box(&mut delta),
                    &sigma,
                    &predecessors,
                    &order,
                );
            });
        });
    }

    group.finish();
}

// Benchmark norm computations
#[cfg(feature = "simd")]
fn bench_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("norms");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let vector = Array1::from_shape_fn(*size, |i| (i as f64).sin());

        group.bench_with_input(BenchmarkId::new("simd_l2", size), size, |b, _| {
            b.iter(|| {
                use scirs2_core::simd_ops::SimdUnifiedOps;
                black_box(f64::simd_norm(&vector.view()))
            });
        });

        group.bench_with_input(BenchmarkId::new("standard_l2", size), size, |b, _| {
            b.iter(|| {
                let norm = vector.dot(&vector).sqrt();
                black_box(norm)
            });
        });

        group.bench_with_input(BenchmarkId::new("simd_l1", size), size, |b, _| {
            b.iter(|| {
                black_box(scirs2_graph::simd_ops::SimdGraphUtils::norm_l1(
                    &vector.view(),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("standard_l1", size), size, |b, _| {
            b.iter(|| {
                let norm: f64 = vector.iter().map(|x| x.abs()).sum();
                black_box(norm)
            });
        });
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(benches, bench_pagerank, bench_laplacian,);

#[cfg(feature = "simd")]
criterion_group!(
    simd_benches,
    bench_matvec,
    bench_power_iteration,
    bench_sparse_matvec,
    bench_betweenness_ops,
    bench_norms,
);

#[cfg(feature = "simd")]
criterion_main!(benches, simd_benches);

#[cfg(not(feature = "simd"))]
criterion_main!(benches);
