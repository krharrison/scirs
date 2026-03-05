//! Advanced Multigrid and Domain Decomposition Solvers for Sparse Systems
//!
//! This module provides production-quality algebraic multigrid (AMG) and
//! domain decomposition preconditioners / solvers for large sparse linear
//! systems arising in scientific computing.
//!
//! # Contents
//!
//! ## Algebraic Multigrid (`algebraic_mg`)
//!
//! Three AMG variants are provided:
//!
//! | Method | Type | Best for |
//! |--------|------|---------|
//! | `RsAmgHierarchy` | Ruge-Stüben AMG | M-matrices, elliptic PDEs |
//! | `SaAmgHierarchy` | Smoothed Aggregation AMG | SPD systems, discretised elasticity |
//! | `AirAmgHierarchy` | AIR AMG | Non-symmetric / advection-dominated |
//!
//! All three support V-cycles, W-cycles, and full iterative solves with
//! convergence monitoring.
//!
//! ## Domain Decomposition (`domain_decomp`)
//!
//! Four domain decomposition methods are provided:
//!
//! | Method | API | Parallelism |
//! |--------|-----|-------------|
//! | Additive Schwarz (ASM) | `additive_schwarz_solve` | Embarrassingly parallel subsolves |
//! | Multiplicative Schwarz (MSM) | `multiplicative_schwarz_solve` | Sequential, SPD convergence guarantee |
//! | FETI | `feti_solve` | Dual interface CG |
//! | Neumann-Neumann Balancing | `neumann_neumann_solve` | Balanced, scalable |
//!
//! # Quick Start
//!
//! ## RS-AMG
//!
//! ```rust
//! use scirs2_sparse::csr::CsrMatrix;
//! use scirs2_sparse::multigrid::algebraic_mg::rs_amg_setup;
//! use scirs2_sparse::iterative_solvers::IterativeSolverConfig;
//!
//! // Build 1D Laplacian
//! let n = 16usize;
//! let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
//! for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
//! for i in 0..n-1 {
//!     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
//!     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
//! }
//! let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
//! let b: Vec<f64> = vec![1.0; n];
//!
//! // Build hierarchy and solve
//! let hier = rs_amg_setup(a).expect("valid input");
//! let config = IterativeSolverConfig { max_iter: 50, tol: 1e-8, verbose: false };
//! let x = hier.solve(&b, &config).expect("valid input");
//! assert_eq!(x.len(), n);
//! ```
//!
//! ## Additive Schwarz
//!
//! ```rust
//! use scirs2_sparse::csr::CsrMatrix;
//! use scirs2_sparse::multigrid::domain_decomp::{additive_schwarz_solve, SchwarzConfig};
//!
//! let n = 8usize;
//! let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
//! for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
//! for i in 0..n-1 {
//!     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
//!     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
//! }
//! let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
//! let b: Vec<f64> = vec![1.0; n];
//! let config = SchwarzConfig { n_subdomains: 2, overlap: 1, max_iter: 100, tol: 1e-8, verbose: false };
//! let result = additive_schwarz_solve(&a, &b, &config).expect("valid input");
//! assert_eq!(result.x.len(), n);
//! ```
//!
//! # References
//!
//! - Ruge & Stüben (1987). "Algebraic multigrid." *Multigrid Methods*, SIAM Frontiers.
//! - Vaněk, Mandel & Brezina (1996). "Algebraic multigrid by smoothed aggregation
//!   for second and fourth order elliptic problems." *Computing*, 56, 179-196.
//! - Manteuffel, Münzenmaier, Ruge & Southworth (2019). "Nonsymmetric reduction-based
//!   algebraic multigrid." *SIAM J. Sci. Comput.*, 41(5), S242-S268.
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*.
//!   Springer.
//! - Farhat & Roux (1991). "A method of finite element tearing and interconnecting."
//!   *Int. J. Numer. Methods Eng.*, 32(6), 1205-1227.
//! - Mandel (1993). "Balancing domain decomposition." *Commun. Numer. Methods Eng.*,
//!   9(3), 233-241.

pub mod algebraic_mg;
pub mod domain_decomp;

// Re-export commonly used types and functions for convenience

// RS-AMG
pub use algebraic_mg::{
    AirAmgHierarchy, AirAmgLevel,
    RsAmgHierarchy, RsAmgLevel,
    SaAmgHierarchy, SaAmgLevel,
    air_amg_setup,
    rs_amg_setup,
    rs_classical_interpolation,
    rs_direct_interpolation,
    sa_amg_setup,
};

// Domain decomposition
pub use domain_decomp::{
    AdditiveSchwarzPreconditioner,
    FetiConfig,
    FetiInterface,
    FetiSolver,
    NeumannNeumannConfig,
    NeumannNeumannSolver,
    SchwarzConfig,
    additive_schwarz_solve,
    feti_solve,
    feti_solve as mg_feti_solve,
    multiplicative_schwarz_solve,
    neumann_neumann_solve,
    partition_overlapping,
};
