//! Differentiable optimization layers (OptNet-style LP/QP).
//!
//! This module implements differentiable quadratic and linear programming
//! layers that can be embedded in gradient-based training pipelines. The
//! backward pass uses implicit differentiation of the KKT conditions to
//! compute gradients of the optimal solution w.r.t. all problem parameters.
//!
//! # Submodules
//!
//! - `kkt_sensitivity`: KKT bordered matrix assembly and adjoint-method sensitivity.
//! - [`qp_layer`]: ADMM-based QP layer with warm-start and active-set backward.
//! - [`lp_layer`]: Entropic LP layer and basis sensitivity analysis.
//! - [`perturbed_optimizer`]: Black-box differentiable combinatorial optimization.
//! - [`implicit_diff`]: Core implicit differentiation engine.
//! - [`combinatorial`]: SparseMAP, soft sort/rank (legacy entry points).
//! - [`diff_qp`]: Interior-point differentiable QP.
//! - [`diff_lp`]: Differentiable LP (active-set based).
//!
//! # References
//! - Amos & Kolter (2017). "OptNet: Differentiable Optimization as a Layer
//!   in Neural Networks." ICML.
//! - Berthet et al. (2020). "Learning with Differentiable Perturbed Optimizers." NeurIPS.
//! - Niculae & Blondel (2017). "A regularized framework for sparse and structured
//!   neural attention." NeurIPS.

pub mod combinatorial;
pub mod diff_lp;
pub mod diff_qp;
pub mod implicit_diff;
pub mod kkt_sensitivity;
pub mod layer;
pub mod lp_layer;
pub mod perturbed_optimizer;
pub mod qp_layer;
pub mod types;

pub use combinatorial::{
    diff_topk, soft_rank, soft_sort, sparsemap, sparsemap_gradient,
    PerturbedOptimizer as PerturbedOptimizerLegacy,
    PerturbedOptimizerConfig as PerturbedOptimizerLegacyConfig, SparsemapConfig, SparsemapResult,
    StructureType,
};
pub use diff_lp::DifferentiableLP;
pub use diff_qp::DifferentiableQP;
pub use kkt_sensitivity::{
    kkt_matrix, kkt_sensitivity, mat_vec, outer_product, parametric_nlp_adjoint, regularize_q,
    sym_outer_product, KktGrad, KktSystem, NlpGrad,
};
pub use layer::{OptNetLayer, StandardOptNetLayer};
pub use lp_layer::{lp_gradient, lp_perturbed, LpLayer, LpLayerConfig, LpSensitivity};
pub use perturbed_optimizer::{
    PerturbedOptimizer, PerturbedOptimizerConfig, SparseMap, SparseMapConfig,
};
pub use qp_layer::{QpLayer, QpLayerConfig};
pub use types::{
    BackwardMode, DiffLPConfig, DiffLPResult, DiffOptGrad, DiffOptParams, DiffOptResult,
    DiffOptStatus, DiffQPConfig, DiffQPResult, ImplicitGradient, KKTSystem,
};
