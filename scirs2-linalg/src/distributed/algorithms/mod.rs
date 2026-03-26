//! Communication-avoiding distributed dense linear algebra algorithms.
//!
//! This module provides implementations of ScaLAPACK-style distributed algorithms
//! for dense linear algebra, including:
//!
//! - **SUMMA**: Scalable Universal Matrix Multiply Algorithm with 2D block-cyclic layout
//! - **CAQR**: Communication-Avoiding QR via Householder with tournament tree reduction
//! - **Distributed Lanczos SVD**: Randomized Lanczos bidiagonalization with thick restart
//!
//! All algorithms operate in a *simulation mode*: a single process models all virtual
//! processors in the grid, making these implementations useful for algorithm validation,
//! cost-model analysis, and portability testing without requiring an actual MPI cluster.
//!
//! # References
//!
//! - Van De Geijn & Watts (1997): *SUMMA: Scalable Universal Matrix Multiply Algorithm*
//! - Demmel et al. (2012): *Communication-optimal parallel and sequential QR and LU factorizations*
//! - Larsen (1998): *Lanczos bidiagonalization with partial reorthogonalization*

pub mod gemm;
pub mod qr;
pub mod svd;

pub use gemm::{BlockCyclicMatrix, CommCost, distributed_gemm_simulate};
pub use qr::{HouseholderReflector, caqr_simulate};
pub use svd::{LanczosSvdConfig, distributed_svd_simulate, thick_restart_lanczos};

use crate::error::LinalgResult;
use scirs2_core::ndarray::Array2;

/// Configuration for distributed linear algebra operations.
///
/// Controls block size and virtual processor grid dimensions used
/// by SUMMA and CAQR simulations.
#[derive(Debug, Clone)]
pub struct DistribConfig {
    /// Tile / block size (rows and columns) used for data partitioning.
    pub block_size: usize,
    /// Number of virtual processor rows in the 2-D process grid.
    pub n_proc_rows: usize,
    /// Number of virtual processor columns in the 2-D process grid.
    pub n_proc_cols: usize,
}

impl Default for DistribConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            n_proc_rows: 2,
            n_proc_cols: 2,
        }
    }
}

/// Trait for distributed dense linear algebra operations.
///
/// Implementors provide distributed GEMM, QR, and SVD that mirror the
/// interface expected by ScaLAPACK-style driver routines.
pub trait DistributedLinearAlgebra {
    /// Distributed general matrix multiply: `C = A * B`.
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand matrix (m × k)
    /// * `b` - Right operand matrix (k × n)
    /// * `config` - Distribution configuration
    ///
    /// # Returns
    ///
    /// Result matrix C (m × n)
    fn distributed_gemm(
        a: &Array2<f64>,
        b: &Array2<f64>,
        config: &DistribConfig,
    ) -> LinalgResult<Array2<f64>>;

    /// Distributed QR decomposition: `A = Q * R`.
    ///
    /// Uses communication-avoiding Householder QR with binary-tree tournament
    /// reduction to achieve O(log P) communication rounds.
    ///
    /// # Returns
    ///
    /// Tuple `(Q, R)` where `Q` is orthogonal and `R` is upper triangular.
    fn distributed_qr(
        a: &Array2<f64>,
        config: &DistribConfig,
    ) -> LinalgResult<(Array2<f64>, Array2<f64>)>;

    /// Distributed truncated SVD via Lanczos bidiagonalization.
    ///
    /// Returns the top-`k` singular triplets of `A`.
    ///
    /// # Returns
    ///
    /// Tuple `(U_k, sigma_k, V_k)` where `sigma_k` are the largest `k` singular
    /// values in descending order.
    fn distributed_svd(
        a: &Array2<f64>,
        k: usize,
    ) -> LinalgResult<(Array2<f64>, Vec<f64>, Array2<f64>)>;
}

/// Concrete implementation that delegates to simulation functions in sub-modules.
pub struct SimulatedDistributed;

impl DistributedLinearAlgebra for SimulatedDistributed {
    fn distributed_gemm(
        a: &Array2<f64>,
        b: &Array2<f64>,
        config: &DistribConfig,
    ) -> LinalgResult<Array2<f64>> {
        distributed_gemm_simulate(a, b, config)
    }

    fn distributed_qr(
        a: &Array2<f64>,
        config: &DistribConfig,
    ) -> LinalgResult<(Array2<f64>, Array2<f64>)> {
        caqr_simulate(a, config)
    }

    fn distributed_svd(
        a: &Array2<f64>,
        k: usize,
    ) -> LinalgResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
        distributed_svd_simulate(a, k)
    }
}
