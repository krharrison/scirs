//! Integration tests for SIMD-accelerated graph operations
//!
//! These tests verify that SIMD and non-SIMD implementations produce
//! identical results across various graph sizes and structures.

use approx::assert_relative_eq;
use scirs2_graph::spectral::LaplacianType;
use scirs2_graph::{cycle_graph, erdos_renyi_graph, path_graph, star_graph, DiGraph, Graph};
use std::collections::HashMap;

const TOLERANCE: f64 = 1e-6;

/// Helper to compare two HashMaps with floating-point values
fn assert_hashmaps_equal<K: Eq + std::hash::Hash + std::fmt::Debug>(
    map1: &HashMap<K, f64>,
    map2: &HashMap<K, f64>,
    tolerance: f64,
) {
    assert_eq!(map1.len(), map2.len(), "HashMaps have different sizes");

    for (key, val1) in map1 {
        let val2 = map2
            .get(key)
            .unwrap_or_else(|| panic!("Key {key:?} not found in second map"));
        assert!(
            (val1 - val2).abs() < tolerance,
            "Values differ: {} vs {}",
            val1,
            val2
        );
    }
}

#[test]
fn test_pagerank_consistency_small_graphs() {
    let graphs = vec![
        ("cycle_5", cycle_graph(5).expect("cycle_graph(5) failed")),
        ("star_6", star_graph(6).expect("star_graph(6) failed")),
        ("path_7", path_graph(7).expect("path_graph(7) failed")),
    ];

    for (name, graph) in graphs {
        // Convert to DiGraph for PageRank
        let mut digraph = DiGraph::<usize, f64>::new();
        for edge in graph.edges() {
            digraph
                .add_edge(edge.source, edge.target, edge.weight)
                .expect("add_edge failed");
            digraph
                .add_edge(edge.target, edge.source, edge.weight)
                .expect("add_edge failed");
        }

        let result1 = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);
        let result2 = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);

        assert_hashmaps_equal(&result1, &result2, TOLERANCE);
        println!("PageRank consistency test passed for {name}");
    }
}

#[test]
fn test_pagerank_consistency_medium_graphs() {
    let sizes = [20, 50, 100];

    for size in sizes {
        let mut rng = scirs2_core::random::seeded_rng(42);
        let graph = erdos_renyi_graph(size, 0.1, &mut rng).expect("erdos_renyi_graph failed");

        // Convert to DiGraph
        let mut digraph = DiGraph::<usize, f64>::new();
        for edge in graph.edges() {
            digraph
                .add_edge(edge.source, edge.target, edge.weight)
                .expect("add_edge failed");
            digraph
                .add_edge(edge.target, edge.source, edge.weight)
                .expect("add_edge failed");
        }

        let result1 = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);
        let result2 = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);

        assert_hashmaps_equal(&result1, &result2, TOLERANCE);

        // Verify PageRank sum is approximately 1.0
        let sum: f64 = result1.values().sum();
        assert_relative_eq!(sum, 1.0, epsilon = TOLERANCE);

        println!("PageRank consistency test passed for size {size}");
    }
}

#[test]
fn test_pagerank_centrality_consistency() {
    let graphs = vec![
        ("cycle_10", cycle_graph(10).expect("cycle_graph(10) failed")),
        ("star_15", star_graph(15).expect("star_graph(15) failed")),
        ("erdos_renyi_20", {
            let mut rng = scirs2_core::random::seeded_rng(123);
            erdos_renyi_graph(20, 0.15, &mut rng).expect("erdos_renyi_graph failed")
        }),
    ];

    for (name, graph) in graphs {
        let result1 = scirs2_graph::pagerank_centrality(&graph, 0.85, 1e-8)
            .expect("pagerank_centrality failed");
        let result2 = scirs2_graph::pagerank_centrality(&graph, 0.85, 1e-8)
            .expect("pagerank_centrality failed");

        assert_hashmaps_equal(&result1, &result2, TOLERANCE);

        // Verify PageRank sum is approximately 1.0
        let sum: f64 = result1.values().sum();
        assert_relative_eq!(sum, 1.0, epsilon = TOLERANCE);

        println!("PageRank centrality consistency test passed for {name}");
    }
}

#[test]
fn test_laplacian_consistency() {
    let graphs = vec![
        ("cycle_8", cycle_graph(8).expect("cycle_graph(8) failed")),
        ("star_10", star_graph(10).expect("star_graph(10) failed")),
        ("path_12", path_graph(12).expect("path_graph(12) failed")),
    ];

    let laplacian_types = vec![
        LaplacianType::Standard,
        LaplacianType::Normalized,
        LaplacianType::RandomWalk,
    ];

    for (name, graph) in graphs {
        for lap_type in &laplacian_types {
            let lap1 = scirs2_graph::laplacian(&graph, *lap_type).expect("laplacian failed");
            let lap2 = scirs2_graph::laplacian(&graph, *lap_type).expect("laplacian failed");

            // Compare matrices element-wise
            assert_eq!(lap1.dim(), lap2.dim());
            for i in 0..lap1.nrows() {
                for j in 0..lap1.ncols() {
                    assert!((lap1[[i, j]] - lap2[[i, j]]).abs() < TOLERANCE);
                }
            }

            println!("Laplacian consistency test passed for {name} ({lap_type:?})");
        }
    }
}

#[test]
fn test_laplacian_properties() {
    let graph = {
        let mut rng = scirs2_core::random::seeded_rng(456);
        erdos_renyi_graph(20, 0.2, &mut rng).expect("erdos_renyi_graph failed")
    };

    // Standard Laplacian
    let lap_std =
        scirs2_graph::laplacian(&graph, LaplacianType::Standard).expect("laplacian failed");

    // Check symmetry
    for i in 0..lap_std.nrows() {
        for j in 0..lap_std.ncols() {
            assert!((lap_std[[i, j]] - lap_std[[j, i]]).abs() < TOLERANCE);
        }
    }

    // Check row sums are zero
    for i in 0..lap_std.nrows() {
        let row_sum: f64 = lap_std.row(i).sum();
        assert!(row_sum.abs() < 1e-10);
    }

    println!("Laplacian properties test passed");
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_vs_standard_pagerank() {
    use scirs2_core::ndarray::Array2;
    use scirs2_graph::simd_ops::SimdPageRank;

    let sizes = [10, 30, 50];

    for size in sizes {
        let graph = cycle_graph(size).expect("cycle_graph failed");

        // Convert to DiGraph and build transition matrix
        let mut digraph = DiGraph::<usize, f64>::new();
        for edge in graph.edges() {
            digraph
                .add_edge(edge.source, edge.target, edge.weight)
                .expect("add_edge failed");
            digraph
                .add_edge(edge.target, edge.source, edge.weight)
                .expect("add_edge failed");
        }

        let n = digraph.node_count();
        let mut transition = Array2::zeros((n, n));
        let out_degrees = digraph.out_degree_vector();

        for (i, node_idx) in digraph.inner().node_indices().enumerate() {
            let node_out_degree = out_degrees[i] as f64;
            if node_out_degree > 0.0 {
                for neighbor_idx in digraph
                    .inner()
                    .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                {
                    let j = neighbor_idx.index();
                    transition[[i, j]] = 1.0 / node_out_degree;
                }
            }
        }

        // SIMD version
        let (simd_result, _) = SimdPageRank::compute_pagerank(&transition, 0.85, 1e-8, 200)
            .expect("compute_pagerank failed");

        // Standard version
        let std_result = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);

        // Compare results
        for (i, (_, rank)) in std_result.iter().enumerate() {
            assert!((simd_result[i] - rank).abs() < 1e-5);
        }

        println!("SIMD vs Standard PageRank test passed for size {size}");
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_vs_standard_laplacian() {
    use scirs2_core::ndarray::Array1;
    use scirs2_graph::simd_ops::SimdSpectral;

    let graphs = vec![
        ("cycle_15", cycle_graph(15).expect("cycle_graph(15) failed")),
        ("star_20", star_graph(20).expect("star_graph(20) failed")),
        ("erdos_renyi_25", {
            let mut rng = scirs2_core::random::seeded_rng(789);
            erdos_renyi_graph(25, 0.15, &mut rng).expect("erdos_renyi_graph failed")
        }),
    ];

    for (name, graph) in graphs {
        let adj_mat = graph.adjacency_matrix();
        let n = graph.node_count();
        let mut adj_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                adj_f64[[i, j]] = adj_mat[[i, j]];
            }
        }

        let degrees = graph.degree_vector();
        let degrees_f64 = Array1::from_shape_fn(n, |i| degrees[i] as f64);

        // Test Standard Laplacian
        let simd_lap_std = SimdSpectral::standard_laplacian(&adj_f64, &degrees_f64)
            .expect("standard_laplacian failed");
        let std_lap_std = scirs2_graph::laplacian(&graph, LaplacianType::Standard)
            .expect("laplacian Standard failed");

        for i in 0..n {
            for j in 0..n {
                assert!((simd_lap_std[[i, j]] - std_lap_std[[i, j]]).abs() < TOLERANCE);
            }
        }

        // Test Normalized Laplacian
        let simd_lap_norm = SimdSpectral::normalized_laplacian(&adj_f64, &degrees_f64)
            .expect("normalized_laplacian failed");
        let std_lap_norm = scirs2_graph::laplacian(&graph, LaplacianType::Normalized)
            .expect("laplacian Normalized failed");

        for i in 0..n {
            for j in 0..n {
                assert!((simd_lap_norm[[i, j]] - std_lap_norm[[i, j]]).abs() < TOLERANCE);
            }
        }

        // Test Random Walk Laplacian
        let simd_lap_rw = SimdSpectral::random_walk_laplacian(&adj_f64, &degrees_f64)
            .expect("random_walk_laplacian failed");
        let std_lap_rw = scirs2_graph::laplacian(&graph, LaplacianType::RandomWalk)
            .expect("laplacian RandomWalk failed");

        for i in 0..n {
            for j in 0..n {
                assert!((simd_lap_rw[[i, j]] - std_lap_rw[[i, j]]).abs() < TOLERANCE);
            }
        }

        println!("SIMD vs Standard Laplacian test passed for {name}");
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_matvec_correctness() {
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_graph::simd_ops::SimdAdjacency;

    let sizes = [5, 10, 25, 50];

    for size in sizes {
        // Create a tridiagonal matrix
        let matrix = Array2::from_shape_fn((size, size), |(i, j)| {
            if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });

        let vector = Array1::from_shape_fn(size, |i| i as f64 + 1.0);

        // SIMD version
        let simd_result =
            SimdAdjacency::dense_matvec(&matrix, &vector).expect("dense_matvec failed");

        // Standard version
        let std_result = matrix.dot(&vector);

        // Compare results
        for i in 0..size {
            assert!((simd_result[i] - std_result[i]).abs() < TOLERANCE);
        }

        println!("SIMD MatVec correctness test passed for size {size}");
    }
}

#[test]
fn test_pagerank_with_dangling_nodes() {
    // Create a graph with a dangling node (no outgoing edges)
    let mut digraph = DiGraph::<usize, f64>::new();
    let _n0 = digraph.add_node(0);
    let _n1 = digraph.add_node(1);
    let _n2 = digraph.add_node(2);

    // 0 -> 1 -> 2, but 2 has no outgoing edges (dangling)
    digraph.add_edge(0, 1, 1.0).expect("add_edge 0->1 failed");
    digraph.add_edge(1, 2, 1.0).expect("add_edge 1->2 failed");

    let result = scirs2_graph::pagerank(&digraph, 0.85, 1e-8, 200);

    // Verify PageRank sum is approximately 1.0
    let sum: f64 = result.values().sum();
    assert_relative_eq!(sum, 1.0, epsilon = TOLERANCE);

    // All nodes should have positive PageRank
    for rank in result.values() {
        assert!(*rank > 0.0);
    }

    println!("PageRank with dangling nodes test passed");
}
