//! Distributed sparse matrix operations.
//!
//! ## Wave 20 (WS115)
//!
//! - [`csr`] — Row-striped CSR partitioning with halo exchange (shared-memory
//!   simulation).
//!
//! ## Wave 28 (WS162)
//!
//! - [`partition`] — Row partitioning (Contiguous / RoundRobin / GraphBased),
//!   [`DistributedCsr`] with ghost-row detection.
//! - [`halo_exchange`] — Simulated halo exchange + parallel distributed SpMV.
//! - [`dist_amg`] — Distributed RS coarsening + direct interpolation AMG
//!   hierarchy and V-cycle solver.

pub mod csr;

pub mod dist_amg;
pub mod halo_exchange;
pub mod partition;

pub use csr::{DistributedCsrConfig, PartitionedCsr};

pub use partition::{
    create_distributed_csr, partition_matrix_nnz, partition_rows, DistributedCsr, PartitionConfig,
    PartitionMethod, RowPartition,
};

pub use halo_exchange::{
    build_halo_messages, distributed_spmv, simulate_halo_exchange, DistributedVector, GhostManager,
    HaloConfig, HaloMessage,
};

pub use dist_amg::{
    build_distributed_amg, build_distributed_interpolation, dist_vcycle, distributed_rs_coarsening,
    DistAMGConfig, DistAMGHierarchy, DistAMGLevel,
};
