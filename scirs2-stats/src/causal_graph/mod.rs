//! Causal Graph Methods
//!
//! This module provides a comprehensive suite of causal inference tools
//! based on Directed Acyclic Graphs (DAGs) and structural causal models.
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`dag`] | [`CausalDAG`] — DAG structure with d-separation (Bayes Ball), Markov blanket, topological sort |
//! | [`identification`] | Do-calculus rules, backdoor/frontdoor criteria, ID algorithm, Tian-Pearl factorisation |
//! | [`estimation`] | IPW, AIPW (doubly robust), nearest-neighbour matching, regression discontinuity, synthetic control, DiD |
//! | [`structure_learning`] | PC algorithm, FCI algorithm, BIC greedy search, LiNGAM, NOTEARS |
//!
//! # Quick Start
//!
//! ## Build and query a DAG
//!
//! ```rust
//! use scirs2_stats::causal_graph::dag::CausalDAG;
//!
//! let mut dag = CausalDAG::new();
//! dag.add_edge("X", "Y").unwrap();
//! dag.add_edge("Y", "Z").unwrap();
//!
//! // D-separation: X ⊥ Z | Y in a chain
//! assert!(dag.is_d_separated("X", "Z", &["Y"]));
//! assert!(!dag.is_d_separated("X", "Z", &[]));
//!
//! let order = dag.topological_sort();
//! assert_eq!(order[0], "X");
//! ```
//!
//! ## Identify P(y | do(x)) using the backdoor criterion
//!
//! ```rust
//! use scirs2_stats::causal_graph::dag::CausalDAG;
//! use scirs2_stats::causal_graph::identification::find_backdoor_sets;
//!
//! // Z → X → Y, Z → Y  (Z is a confounder)
//! let mut dag = CausalDAG::new();
//! dag.add_edge("Z", "X").unwrap();
//! dag.add_edge("Z", "Y").unwrap();
//! dag.add_edge("X", "Y").unwrap();
//!
//! let bd = find_backdoor_sets(&dag, "X", "Y", 3);
//! assert!(bd.is_admissible);
//! assert!(bd.adjustment_set.contains(&"Z".to_string()));
//! ```
//!
//! ## Estimate ATE with IPW
//!
//! ```rust
//! use scirs2_stats::causal_graph::estimation::IPWEstimator;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let n = 20;
//! let cov = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
//! let treat = Array1::from_iter((0..n).map(|i| if i < n/2 { 1.0 } else { 0.0 }));
//! let outcome = Array1::from_iter((0..n).map(|i| if i < n/2 { 2.0 } else { 0.0 }));
//!
//! let est = IPWEstimator::default();
//! let result = est.estimate(cov.view(), treat.view(), outcome.view()).unwrap();
//! assert!(result.estimate.is_finite());
//! ```
//!
//! ## Learn causal structure from data
//!
//! ```rust
//! use scirs2_stats::causal_graph::structure_learning::PcAlgorithm;
//! use scirs2_core::ndarray::Array2;
//!
//! let n = 50;
//! let data = Array2::<f64>::from_shape_fn((n, 3), |(i, j)| {
//!     (i as f64 * 0.1 + j as f64).sin()
//! });
//! let pc = PcAlgorithm::default();
//! let result = pc.fit(data.view(), &["A", "B", "C"]).unwrap();
//! assert_eq!(result.dag.n_nodes(), 3);
//! ```
//!
//! # References
//!
//! - Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*.
//!   Cambridge University Press.
//! - Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction,
//!   and Search* (2nd ed.). MIT Press.
//! - Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics,
//!   Social, and Biomedical Sciences*. Cambridge University Press.
//! - Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
//!   Distributions in Recursive Semi-Markovian Causal Models. AAAI 2006.
//! - Zheng, X. et al. (2018). DAGs with NO TEARS. NeurIPS 2018.
//! - Shimizu, S. et al. (2006). A Linear Non-Gaussian Acyclic Model for
//!   Causal Discovery. JMLR.

pub mod dag;
pub mod estimation;
pub mod identification;
pub mod structure_learning;

// ---------------------------------------------------------------------------
// Re-exports — DAG
// ---------------------------------------------------------------------------

pub use dag::CausalDAG;

// ---------------------------------------------------------------------------
// Re-exports — Identification
// ---------------------------------------------------------------------------

pub use identification::{
    check_do_calculus_rule,
    c_components_with_hidden,
    find_backdoor_sets,
    find_frontdoor_set,
    id_algorithm,
    satisfies_backdoor,
    satisfies_frontdoor,
    tian_pearl_id,
    BackdoorResult,
    CComponent,
    DoCalculusRule,
    FrontdoorResult,
    IdResult,
};

// ---------------------------------------------------------------------------
// Re-exports — Estimation
// ---------------------------------------------------------------------------

pub use estimation::{
    DifferenceInDifferences,
    DiDResult,
    DoublyRobustEstimator,
    EstimationResult,
    IPWEstimator,
    NearestNeighborMatching,
    RegressionDiscontinuity,
    SyntheticControlEstimator,
    SyntheticControlResult,
};

// ---------------------------------------------------------------------------
// Re-exports — Structure Learning
// ---------------------------------------------------------------------------

pub use structure_learning::{
    BicGreedySearch,
    EdgeType,
    FciAlgorithm,
    LiNGAM,
    LiNGAMResult,
    Notears,
    PcAlgorithm,
    StructureLearningResult,
};
