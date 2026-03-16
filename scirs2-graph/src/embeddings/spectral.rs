//! Functional spectral embedding API for graphs
//!
//! Provides simple function-based wrappers around the `SpectralEmbedding` struct
//! in `spectral_embedding.rs`.  These functions produce `Vec<Vec<f64>>` embeddings
//! indexed by node position in `graph.nodes()`.
//!
//! # Functions
//! - `spectral_embedding`             — Laplacian eigenmap embedding
//! - `adjacency_spectral_embedding`   — adjacency matrix eigendecomposition
//!
//! Both functions accept a `Graph<usize, f64>` and return one row per node in
//! the order that `graph.nodes()` yields them (which matches node insertion order
//! when nodes are `usize` indices 0, 1, …, n-1).

use super::spectral_embedding::{SpectralEmbedding, SpectralEmbeddingConfig, SpectralLaplacianType};
use crate::base::{Graph, Node, EdgeWeight};
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// spectral_embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a Laplacian eigenmap embedding (spectral embedding) of the graph.
///
/// Uses the Ng-Jordan-Weiss normalised Laplacian by default.  The embedding
/// maps each node to a `dim`-dimensional vector that reflects the spectral
/// geometry of the graph; nearby nodes in the embedding correspond to nodes
/// that are densely connected.
///
/// # Arguments
/// * `graph` – the undirected graph
/// * `dim`   – number of embedding dimensions
///
/// # Returns
/// `Vec<Vec<f64>>` of length `graph.node_count()` where `result[i]` is the
/// embedding of the `i`-th node as returned by `graph.nodes()`.
/// Returns an error if the graph is empty or `dim` is too large.
pub fn spectral_embedding<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    dim: usize,
) -> Result<Vec<Vec<f64>>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + scirs2_core::numeric::Zero + scirs2_core::numeric::One + Copy,
    Ix: petgraph::graph::IndexType,
{
    let config = SpectralEmbeddingConfig {
        dimensions: dim,
        laplacian_type: SpectralLaplacianType::NormalizedNgJordanWeiss,
        normalize: true,
        drop_first: true,
        ..Default::default()
    };

    let mut se = SpectralEmbedding::new(config);
    se.fit(graph)?;

    let emb_map = se.embeddings()?;
    let nodes = graph.nodes();

    let mut result = Vec::with_capacity(nodes.len());
    for node in nodes {
        let emb = emb_map
            .get(node)
            .ok_or_else(|| GraphError::AlgorithmError(format!("Missing embedding for node")))?;
        result.push(emb.vector.clone());
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// adjacency_spectral_embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an adjacency-matrix spectral embedding of the graph.
///
/// Uses the top `dim` eigenvectors of the (weighted) adjacency matrix.  This
/// differs from the Laplacian eigenmap approach: similar nodes that are
/// *directly connected* will have similar embedding vectors, whereas the
/// Laplacian approach groups nodes that are in the same "community" even if
/// not directly linked.
///
/// Internally this uses the unnormalized Laplacian with `drop_first = false`
/// (so the first eigenvector, corresponding to the adjacency eigenvector of
/// the largest eigenvalue, is retained).
///
/// # Arguments
/// * `graph` – the undirected graph
/// * `dim`   – number of embedding dimensions
///
/// # Returns
/// `Vec<Vec<f64>>` of length `graph.node_count()`.
pub fn adjacency_spectral_embedding<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    dim: usize,
) -> Result<Vec<Vec<f64>>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: petgraph::graph::IndexType,
{
    // Use the adjacency matrix directly: compute its top-k eigenvectors using
    // power iteration / deflation.
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "Cannot compute adjacency spectral embedding for empty graph".to_string(),
        ));
    }
    if dim == 0 {
        return Err(GraphError::InvalidParameter {
            param: "dim".to_string(),
            value: "0".to_string(),
            expected: "dim >= 1".to_string(),
            context: "adjacency_spectral_embedding".to_string(),
        });
    }
    let effective_dim = dim.min(n);

    // Build adjacency matrix as Vec<Vec<f64>>
    let nodes = graph.nodes();
    let node_to_idx: HashMap<*const N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| ((*n) as *const N, i))
        .collect();

    let edges = graph.edges();
    let mut adj = vec![vec![0.0f64; n]; n];
    for e in &edges {
        let si = node_to_idx
            .get(&((&e.source) as *const N))
            .copied();
        let ti = node_to_idx
            .get(&((&e.target) as *const N))
            .copied();
        if let (Some(s), Some(t)) = (si, ti) {
            let w: f64 = e.weight.clone().into();
            adj[s][t] += w;
            adj[t][s] += w;
        }
    }

    // Power iteration to find top-k eigenvectors of adj
    let mut rng = scirs2_core::random::rng();
    let eigvecs = power_iteration_top_k(&adj, n, effective_dim, &mut rng);

    Ok(eigvecs)
}

/// Power-iteration with deflation to find the top-k eigenvectors of a
/// symmetric matrix (stored as `Vec<Vec<f64>>`).
///
/// Returns a `Vec<Vec<f64>>` where `result[i]` is the embedding of node `i`.
fn power_iteration_top_k(mat: &[Vec<f64>], n: usize, k: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    let max_iter = 200usize;
    let tol = 1e-8;

    // Working copy of the matrix (we deflate each eigenvector out)
    let mut working: Vec<Vec<f64>> = mat.to_vec();

    // Eigenvectors: eigvecs[j][i] = component of j-th eigvec for node i
    let mut eigvecs: Vec<Vec<f64>> = Vec::with_capacity(k);

    for _ in 0..k {
        // Random initialisation
        let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
        normalise_vec(&mut v);

        let mut prev_eigenvalue = 0.0;

        for _ in 0..max_iter {
            let mut w = mat_vec_mul(&working, &v, n);
            let eigenvalue = dot_vec(&v, &w, n);
            normalise_vec(&mut w);

            // Orthogonalise against previously found eigenvectors
            for ev in &eigvecs {
                let proj = dot_vec(ev, &w, n);
                for i in 0..n {
                    w[i] -= proj * ev[i];
                }
            }
            normalise_vec(&mut w);

            if (eigenvalue - prev_eigenvalue).abs() < tol {
                v = w;
                break;
            }
            prev_eigenvalue = eigenvalue;
            v = w;
        }

        // Deflate: working = working - eigenvalue * v * v^T
        let eigenvalue = dot_vec(&v, &mat_vec_mul(&working, &v, n), n);
        for i in 0..n {
            for j in 0..n {
                working[i][j] -= eigenvalue * v[i] * v[j];
            }
        }

        eigvecs.push(v);
    }

    // Transpose: result[node] = [eigvec_0[node], eigvec_1[node], ...]
    let mut result: Vec<Vec<f64>> = vec![Vec::with_capacity(k); n];
    for ev in &eigvecs {
        for i in 0..n {
            result[i].push(ev[i]);
        }
    }
    result
}

#[inline]
fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += mat[i][j] * v[j];
        }
    }
    result
}

#[inline]
fn dot_vec(a: &[f64], b: &[f64], n: usize) -> f64 {
    a.iter().take(n).zip(b.iter().take(n)).map(|(x, y)| x * y).sum()
}

#[inline]
fn normalise_vec(v: &mut Vec<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path_graph() -> Graph<usize, f64> {
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        for i in 0..4 {
            let _ = g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    fn make_complete_graph() -> Graph<usize, f64> {
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    #[test]
    fn test_spectral_embedding_dimensions() {
        let g = make_path_graph();
        let result = spectral_embedding(&g, 2).expect("spectral embedding should succeed");
        assert_eq!(result.len(), 5, "should have one row per node");
        for row in &result {
            assert_eq!(row.len(), 2, "each row should have 2 dimensions");
        }
    }

    #[test]
    fn test_spectral_embedding_unit_norms() {
        let g = make_path_graph();
        let result = spectral_embedding(&g, 2).expect("should succeed");
        // With normalize=true, rows should be roughly unit length
        for row in &result {
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                norm <= 1.0 + 1e-6,
                "embedding vector norm should be ≤ 1.0, got {norm}"
            );
        }
    }

    #[test]
    fn test_spectral_embedding_empty_graph_error() {
        let g: Graph<usize, f64> = Graph::new();
        let result = spectral_embedding(&g, 2);
        assert!(result.is_err(), "empty graph should return error");
    }

    #[test]
    fn test_adjacency_spectral_embedding_dimensions() {
        let g = make_path_graph();
        let result = adjacency_spectral_embedding(&g, 2).expect("should succeed");
        assert_eq!(result.len(), 5);
        for row in &result {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_adjacency_spectral_embedding_complete_graph() {
        let g = make_complete_graph();
        let result = adjacency_spectral_embedding(&g, 2).expect("should succeed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_adjacency_spectral_embedding_empty_error() {
        let g: Graph<usize, f64> = Graph::new();
        let result = adjacency_spectral_embedding(&g, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_adjacency_spectral_embedding_dim_zero_error() {
        let g = make_path_graph();
        let result = adjacency_spectral_embedding(&g, 0);
        assert!(result.is_err());
    }
}
