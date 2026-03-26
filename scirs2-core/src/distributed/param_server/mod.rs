//! Fault-tolerant parameter server for distributed computing
//!
//! This module provides a parameter server implementation with support for
//! multiple consistency models (BSP, ASP, SSP), gossip-based all-reduce,
//! and fault tolerance with checkpointing and heartbeat monitoring.

pub mod fault_tolerance;
pub mod gossip;
pub mod server;
pub mod types;

pub use fault_tolerance::{CheckpointConfig, FaultTolerantPS, FaultTolerantPs, VectorClock};
pub use gossip::{gossip_allreduce_simulate, GossipAllReduce};
pub use server::{ParameterServer, ServerCheckpoint};
pub use types::{
    AggregationMethod, ConsistencyModel, GossipConfig, GossipTopology, ParamServerConfig,
    ParameterKey, ParameterUpdate, WorkerState,
};
