//! Conic programming: SDP and SOCP solvers.
//!
//! Provides interior-point methods for:
//! - Semidefinite Programming (SDP) via Mehrotra predictor-corrector
//! - Second-Order Cone Programming (SOCP) via NT-scaling interior point

pub mod sdp;
pub mod socp;

pub use sdp::{
    MaxCutSdpResult, MatrixCompletionSdpResult, SDPProblem, SDPResult,
    SDPSolver, SDPSolverConfig, max_cut_sdp, matrix_completion_sdp,
};
pub use socp::{
    PortfolioSocpResult, RobustLsResult, SOCConstraint, SOCPConfig,
    SOCPProblem, SOCPResult, portfolio_optimization_socp, robust_ls_socp,
    socp_to_sdp,
};
