//! Linear algebra operations for sparse matrices
//!
//! This module provides linear algebra operations for sparse matrices,
//! including solvers, eigenvalue computations, and matrix functions.

mod amg;
// mod banded_solvers; // Will be added separately
mod cgs;
mod decomposition;
mod eigen;
mod enhanced_operators;
mod expm;
mod gcrot;
mod ic;
mod interface;
mod iterative;
mod lgmres;
mod lsmr;
mod lsqr;
mod matfuncs;
mod minres;
mod preconditioners;
mod qmr;
mod qmr_simple;
mod solvers;
mod spai;
mod specialized_solvers; // v0.2.0
mod svd;
mod tfqmr;

pub use amg::{AMGOptions, AMGPreconditioner, CycleType, InterpolationType, SmootherType};
// pub use banded_solvers::*; // Will be added separately
pub use cgs::{cgs, CGSOptions, CGSResult};
pub use decomposition::{
    cholesky_decomposition, incomplete_cholesky, incomplete_lu, ldlt_decomposition,
    lu_decomposition, lu_decomposition_with_options, pivoted_cholesky_decomposition,
    qr_decomposition, CholeskyResult, ICOptions, ILUOptions, LDLTResult, LUOptions, LUResult,
    PivotedCholeskyResult, PivotingStrategy, QRResult,
};
pub use eigen::{
    eigs, eigsh, eigsh_generalized, eigsh_generalized_enhanced, eigsh_shift_invert,
    eigsh_shift_invert_enhanced, iram, iram_shift_invert, lanczos, power_iteration, ArnoldiConfig,
    ArpackOptions, EigenResult, EigenvalueMethod, EigenvalueMode, LanczosOptions,
    PowerIterationOptions,
};
pub use enhanced_operators::{
    convolution_operator, enhanced_add, enhanced_diagonal, enhanced_scale, enhanced_subtract,
    finite_difference_operator, BoundaryCondition, ConvolutionMode, ConvolutionOperator,
    EnhancedDiagonalOperator, EnhancedDifferenceOperator, EnhancedOperatorOptions,
    EnhancedScaledOperator, EnhancedSumOperator, FiniteDifferenceOperator,
};
pub use expm::expm;
pub use gcrot::{gcrot, GCROTOptions, GCROTResult};
pub use ic::IC0Preconditioner;
pub use interface::{
    AsLinearOperator, DiagonalOperator, IdentityOperator, LinearOperator, ScaledIdentityOperator,
};
pub use iterative::{
    bicg, bicgstab, cg, gmres, BiCGOptions, BiCGSTABOptions, BiCGSTABResult, CGOptions,
    GMRESOptions, IterationResult, IterativeSolver,
};
pub use lgmres::{lgmres, LGMRESOptions, LGMRESResult};
pub use lsmr::{lsmr, LSMROptions, LSMRResult};
pub use lsqr::{lsqr, LSQROptions, LSQRResult};
pub use matfuncs::{
    condest, condest_enhanced, expm_multiply, onenormest, onenormest_enhanced, twonormest,
    twonormest_enhanced,
};
pub use minres::{minres, MINRESOptions, MINRESResult};
pub use preconditioners::{ILU0Preconditioner, JacobiPreconditioner, SSORPreconditioner};
pub use qmr::{qmr, QMROptions, QMRResult};
pub use solvers::{
    add, diag_matrix, eye, inv, matmul, matrix_power, multiply, norm, sparse_direct_solve,
    sparse_lstsq, spsolve,
};
pub use spai::{SpaiOptions, SpaiPreconditioner};
pub use specialized_solvers::{
    solve_arrow_matrix, solve_banded_system, solve_block_2x2, solve_kronecker_system,
    solve_saddle_point,
}; // v0.2.0
pub use svd::{svd_truncated, svds, SVDOptions, SVDResult};
pub use tfqmr::{tfqmr, TFQMROptions, TFQMRResult};
