//! Probabilistic Graphical Models (PGM)
//!
//! This module provides implementations of:
//! - Bayesian Networks (directed graphical models with exact and approximate inference)
//! - Markov Random Fields (undirected graphical models with Gibbs sampling and belief propagation)
//! - Factor Graphs (general graphical models with sum-product and max-product algorithms)
//!
//! # Example: Bayesian Network
//! ```
//! use scirs2_stats::pgm::bayesian_network::{BayesianNetwork, ConditionalProbability};
//! use std::collections::HashMap;
//!
//! let mut bn = BayesianNetwork::new();
//! bn.add_node("Rain", 2).unwrap();
//! bn.add_node("WetGrass", 2).unwrap();
//! bn.add_edge("Rain", "WetGrass").unwrap();
//! ```

pub mod bayesian_network;
pub mod factor_graph;
pub mod markov_random_field;

pub use bayesian_network::*;
pub use factor_graph::*;
pub use markov_random_field::*;
