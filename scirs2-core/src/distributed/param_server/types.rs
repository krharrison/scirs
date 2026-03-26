//! Types for the fault-tolerant parameter server

/// Configuration for the parameter server
#[derive(Debug, Clone)]
pub struct ParamServerConfig {
    /// Number of workers participating
    pub num_workers: usize,
    /// Consistency model to use
    pub consistency: ConsistencyModel,
    /// Method for aggregating updates from workers
    pub aggregation: AggregationMethod,
    /// How often to checkpoint (in steps)
    pub checkpoint_interval: usize,
    /// Number of parameter replicas for fault tolerance
    pub replication_factor: usize,
}

impl Default for ParamServerConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            consistency: ConsistencyModel::BSP,
            aggregation: AggregationMethod::Mean,
            checkpoint_interval: 100,
            replication_factor: 1,
        }
    }
}

/// Consistency model for parameter updates
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConsistencyModel {
    /// Bulk Synchronous Parallel: all workers must push before updates are applied
    BSP,
    /// Asynchronous Parallel: updates are applied immediately
    ASP,
    /// Stale Synchronous Parallel: updates are applied if within staleness bound
    SSP {
        /// Maximum allowed staleness (version difference between fastest and slowest worker)
        max_staleness: usize,
    },
}

/// Method for aggregating parameter updates from multiple workers
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AggregationMethod {
    /// Arithmetic mean of all updates
    Mean,
    /// Sum of all updates
    Sum,
    /// Weighted mean using per-worker weights
    WeightedMean {
        /// Weight for each worker (indexed by worker_id)
        weights: Vec<f64>,
    },
}

/// Configuration for gossip-based all-reduce
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Network topology for gossip communication
    pub topology: GossipTopology,
    /// Maximum number of gossip rounds
    pub num_rounds: usize,
    /// Fraction of value to push to neighbor (0.0 to 1.0)
    pub push_fraction: f64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            topology: GossipTopology::Ring,
            num_rounds: 10,
            push_fraction: 0.5,
        }
    }
}

/// Network topology for gossip protocol
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GossipTopology {
    /// Ring topology: each node sends to (id + 1) % n
    Ring,
    /// Random topology: each node picks a uniform random peer
    Random,
    /// Exponential topology: each node sends to id + 2^k for random k
    Exponential,
}

/// A parameter update from a worker
#[derive(Debug, Clone)]
pub struct ParameterUpdate {
    /// Parameter key being updated
    pub key: String,
    /// Updated values
    pub values: Vec<f64>,
    /// ID of the worker sending the update
    pub worker_id: usize,
    /// Version of the update (worker-local iteration counter)
    pub version: u64,
}

/// A key for looking up parameters
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParameterKey {
    /// The string key identifying the parameter
    pub key: String,
}

impl ParameterKey {
    /// Create a new parameter key
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self { key: key.into() }
    }
}

/// State of a worker in the parameter server
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Unique ID of this worker
    pub worker_id: usize,
    /// Current version (iteration) of this worker
    pub version: u64,
    /// Whether the worker is alive
    pub is_alive: bool,
    /// Timestamp of last heartbeat
    pub last_heartbeat: u64,
}

impl WorkerState {
    /// Create a new worker state with the given ID
    #[must_use]
    pub fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            version: 0,
            is_alive: true,
            last_heartbeat: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParamServerConfig::default();
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.consistency, ConsistencyModel::BSP);
        assert_eq!(config.checkpoint_interval, 100);
        assert_eq!(config.replication_factor, 1);
    }

    #[test]
    fn test_gossip_config_default() {
        let config = GossipConfig::default();
        assert_eq!(config.topology, GossipTopology::Ring);
        assert_eq!(config.num_rounds, 10);
        assert!((config.push_fraction - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parameter_key() {
        let key = ParameterKey::new("layer1.weight");
        assert_eq!(key.key, "layer1.weight");
    }

    #[test]
    fn test_worker_state_new() {
        let ws = WorkerState::new(3);
        assert_eq!(ws.worker_id, 3);
        assert_eq!(ws.version, 0);
        assert!(ws.is_alive);
        assert_eq!(ws.last_heartbeat, 0);
    }
}
