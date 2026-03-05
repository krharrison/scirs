//! Standalone network flow algorithms (integer-capacity, index-based API).
//!
//! This module provides high-performance, standalone flow algorithms that
//! operate on integer capacities and node indices rather than the generic
//! `Graph` type.  They are used internally by other modules (e.g. hypergraph
//! connectivity) and can be used directly for competitive-programming-style
//! graph problems.
//!
//! ## Algorithms
//!
//! - [`max_flow::DinicMaxFlow`] -- Dinic's algorithm, O(V^2 E)
//! - [`max_flow::PushRelabelMaxFlow`] -- Push-Relabel, O(V^2 sqrt(E))
//! - [`max_flow::EdmondsKarp`] -- Edmonds-Karp (BFS Ford-Fulkerson), O(V E^2)
//! - [`min_cost_flow::MinCostFlow`] -- SPFA successive shortest paths
//! - [`min_cost_flow::PotentialMinCostFlow`] -- Johnson potentials + Dijkstra

pub mod max_flow;
pub mod min_cost_flow;
