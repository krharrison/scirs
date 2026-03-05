//! Bayesian Networks and Probabilistic Graphical Models.
//!
//! This module provides a complete implementation of Bayesian Networks with:
//!
//! ## Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`dag`] | Index-based DAG with d-separation, topological sort, Markov blanket, v-structures |
//! | [`cpd`] | Conditional probability distributions: tabular, Gaussian, mixture, conditional linear |
//! | [`exact_inference`] | Variable elimination and belief propagation |
//! | [`approximate_inference`] | Gibbs sampling, likelihood weighting, mean-field VI |
//! | [`structure_learning`] | PC algorithm, hill climbing, BIC score |
//!
//! ## Quick Start
//!
//! ### Build and query a Bayesian Network
//!
//! ```rust
//! use scirs2_stats::bayesian_network::{
//!     dag::DAG,
//!     cpd::{CPD, TabularCPD},
//!     exact_inference::{BayesianNetwork, VariableElimination},
//! };
//! use std::collections::HashMap;
//!
//! // Build a Rain → WetGrass ← Sprinkler network
//! let mut dag = DAG::new(3);
//! dag.add_edge(0, 2).unwrap(); // Rain → WetGrass
//! dag.add_edge(1, 2).unwrap(); // Sprinkler → WetGrass
//!
//! // P(Rain)
//! let cpd_rain = TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.8, 0.2]]).unwrap();
//! // P(Sprinkler)
//! let cpd_spr = TabularCPD::new(1, 2, vec![], vec![], vec![vec![0.5, 0.5]]).unwrap();
//! // P(WetGrass | Rain, Sprinkler)
//! let cpd_wg = TabularCPD::new(
//!     2, 2, vec![0, 1], vec![2, 2],
//!     vec![
//!         vec![0.99, 0.01], vec![0.01, 0.99],
//!         vec![0.01, 0.99], vec![0.01, 0.99],
//!     ],
//! ).unwrap();
//!
//! let cpds: Vec<Box<dyn CPD>> = vec![
//!     Box::new(cpd_rain), Box::new(cpd_spr), Box::new(cpd_wg),
//! ];
//! let bn = BayesianNetwork::new(dag, cpds).unwrap();
//!
//! // Query P(Rain | WetGrass = 1)
//! let mut evidence = HashMap::new();
//! evidence.insert(2usize, 1usize); // WetGrass = 1 (wet)
//!
//! let ve = VariableElimination::from_network(&bn, &[0], &evidence);
//! let result = ve.query(&bn, &[0], &evidence).unwrap();
//! // P(Rain=1 | WetGrass=1) > 0.2 (prior)
//! assert!(result[&0][1] > 0.2);
//! ```
//!
//! ### D-separation test
//!
//! ```rust
//! use scirs2_stats::bayesian_network::dag::DAG;
//!
//! // Chain: 0 → 1 → 2
//! let mut dag = DAG::new(3);
//! dag.add_edge(0, 1).unwrap();
//! dag.add_edge(1, 2).unwrap();
//!
//! // 0 ⊥ 2 | 1 in a chain
//! assert!(dag.d_separation(0, 2, &[1]));
//! assert!(!dag.d_separation(0, 2, &[]));
//! ```
//!
//! ## References
//!
//! - Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*.
//!   MIT Press.
//! - Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*.
//!   Morgan Kaufmann.
//! - Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction,
//!   and Search* (2nd ed.). MIT Press.

pub mod approximate_inference;
pub mod cpd;
pub mod dag;
pub mod exact_inference;
pub mod structure_learning;

// ---------------------------------------------------------------------------
// Top-level re-exports for convenience
// ---------------------------------------------------------------------------

pub use dag::DAG;
pub use cpd::{CPD, TabularCPD, GaussianCPD, MixtureCPD, ConditionalLinear};
pub use exact_inference::{BayesianNetwork, Factor, VariableElimination, BeliefPropagation};
pub use approximate_inference::{GibbsSampler, LikelihoodWeighting, MeanFieldVI, Rng, LcgRng};
pub use structure_learning::{PCAlgorithm, HillClimbing, BIC, Operator, fisherz_test, partial_correlation, count_cardinalities};
