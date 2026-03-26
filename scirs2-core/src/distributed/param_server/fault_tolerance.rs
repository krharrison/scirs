//! Fault tolerance for the parameter server
//!
//! Provides heartbeat monitoring, worker failure detection, automatic
//! checkpointing, and vector clocks for causal ordering.

use crate::error::{CoreError, CoreResult, ErrorContext};

use super::server::{ParameterServer, ServerCheckpoint};
use super::types::ParamServerConfig;

/// Configuration for the checkpointing subsystem
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// How often to checkpoint (in steps); 0 disables auto-checkpointing
    pub checkpoint_every: usize,
    /// Maximum number of rolling checkpoints to retain
    pub max_checkpoints: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_every: 10,
            max_checkpoints: 3,
        }
    }
}

/// Fault-tolerant PS with rolling checkpoints and heartbeat-based failure detection
///
/// This is a richer companion to [`FaultTolerantPS`] that keeps multiple rolling
/// checkpoints and integrates a millisecond-granularity heartbeat timeout.
#[derive(Debug)]
pub struct FaultTolerantPs {
    /// The underlying parameter server
    server: ParameterServer,
    /// Checkpoint configuration
    checkpoint_config: CheckpointConfig,
    /// Rolling checkpoint store (newest last)
    checkpoints: std::collections::VecDeque<ServerCheckpoint>,
    /// Per-worker heartbeat timestamp (as Instant)
    worker_heartbeats: std::collections::HashMap<usize, std::time::Instant>,
    /// Heartbeat timeout in milliseconds
    heartbeat_timeout_ms: u64,
}

impl FaultTolerantPs {
    /// Create a new `FaultTolerantPs` wrapping `server`
    #[must_use]
    pub fn new(server: ParameterServer, checkpoint_config: CheckpointConfig) -> Self {
        let n = server.num_workers();
        let mut worker_heartbeats = std::collections::HashMap::new();
        for i in 0..n {
            worker_heartbeats.insert(i, std::time::Instant::now());
        }
        Self {
            server,
            checkpoint_config,
            checkpoints: std::collections::VecDeque::new(),
            worker_heartbeats,
            heartbeat_timeout_ms: 5000,
        }
    }

    /// Get a reference to the underlying server
    #[must_use]
    pub fn server(&self) -> &ParameterServer {
        &self.server
    }

    /// Get a mutable reference to the underlying server
    pub fn server_mut(&mut self) -> &mut ParameterServer {
        &mut self.server
    }

    /// Take a checkpoint, rolling off the oldest if `max_checkpoints` is exceeded
    pub fn checkpoint(&mut self) -> CoreResult<()> {
        let cp = self.server.checkpoint();
        self.checkpoints.push_back(cp);
        while self.checkpoints.len() > self.checkpoint_config.max_checkpoints {
            self.checkpoints.pop_front();
        }
        Ok(())
    }

    /// Restore the parameter server from the most recent checkpoint
    pub fn restore_latest(&mut self) -> CoreResult<()> {
        let cp = self.checkpoints.back().ok_or_else(|| {
            CoreError::ComputationError(ErrorContext::new(
                "No checkpoint available to restore".to_string(),
            ))
        })?;
        let config = self.server.config().clone();
        self.server = ParameterServer::restore(cp, config)?;
        Ok(())
    }

    /// Record a heartbeat for `worker_id` (updates timestamp to now)
    pub fn heartbeat(&mut self, worker_id: usize) {
        self.worker_heartbeats
            .insert(worker_id, std::time::Instant::now());
    }

    /// Return the IDs of workers that have missed their heartbeat deadline
    #[must_use]
    pub fn detect_failed_workers(&self) -> Vec<usize> {
        let timeout = std::time::Duration::from_millis(self.heartbeat_timeout_ms);
        let now = std::time::Instant::now();
        let mut failed = Vec::new();
        for (worker_id, last_beat) in &self.worker_heartbeats {
            if now.duration_since(*last_beat) > timeout {
                failed.push(*worker_id);
            }
        }
        failed.sort_unstable();
        failed
    }

    /// Return the number of retained checkpoints
    #[must_use]
    pub fn n_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Set heartbeat timeout in milliseconds
    pub fn set_heartbeat_timeout_ms(&mut self, ms: u64) {
        self.heartbeat_timeout_ms = ms;
    }
}

/// Fault-tolerant wrapper around `ParameterServer`
///
/// Adds heartbeat monitoring, failure detection, automatic checkpointing,
/// and recovery capabilities.
#[derive(Debug)]
pub struct FaultTolerantPS {
    /// The underlying parameter server
    server: ParameterServer,
    /// Last checkpoint taken
    last_checkpoint: Option<ServerCheckpoint>,
    /// Step at which last checkpoint was taken
    last_checkpoint_step: usize,
}

impl FaultTolerantPS {
    /// Create a new fault-tolerant parameter server
    #[must_use]
    pub fn new(config: ParamServerConfig) -> Self {
        Self {
            server: ParameterServer::new(config),
            last_checkpoint: None,
            last_checkpoint_step: 0,
        }
    }

    /// Get a reference to the underlying parameter server
    #[must_use]
    pub fn server(&self) -> &ParameterServer {
        &self.server
    }

    /// Get a mutable reference to the underlying parameter server
    pub fn server_mut(&mut self) -> &mut ParameterServer {
        &mut self.server
    }

    /// Record a heartbeat from a worker
    pub fn heartbeat(&mut self, worker_id: usize, timestamp: u64) -> CoreResult<()> {
        let workers = self.server.workers_mut();
        if worker_id >= workers.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Unknown worker ID: {worker_id}"
            ))));
        }
        workers[worker_id].last_heartbeat = timestamp;
        workers[worker_id].is_alive = true;
        Ok(())
    }

    /// Check for timed-out workers
    ///
    /// Returns the IDs of workers whose last heartbeat is older than
    /// `current_time - timeout`.
    #[must_use]
    pub fn check_workers(&self, current_time: u64, timeout: u64) -> Vec<usize> {
        self.server
            .workers()
            .iter()
            .filter(|w| w.is_alive && current_time.saturating_sub(w.last_heartbeat) > timeout)
            .map(|w| w.worker_id)
            .collect()
    }

    /// Handle a worker failure by marking it as dead
    ///
    /// Any pending updates from this worker in the buffer are discarded
    /// during the next barrier sync (BSP) since the worker count is reduced.
    pub fn handle_worker_failure(&mut self, failed_worker: usize) -> CoreResult<()> {
        let workers = self.server.workers_mut();
        if failed_worker >= workers.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Unknown worker ID: {failed_worker}"
            ))));
        }
        if !workers[failed_worker].is_alive {
            return Err(CoreError::ComputationError(ErrorContext::new(format!(
                "Worker {failed_worker} is already marked as dead"
            ))));
        }
        workers[failed_worker].is_alive = false;
        Ok(())
    }

    /// Check if a checkpoint should be taken at this step, and if so, take it
    ///
    /// Returns the checkpoint if one was taken.
    pub fn checkpoint_if_needed(&mut self, step: usize) -> Option<ServerCheckpoint> {
        let interval = self.server.config().checkpoint_interval;
        if interval == 0 {
            return None;
        }
        if step > 0 && step % interval == 0 && step > self.last_checkpoint_step {
            let cp = self.server.checkpoint();
            self.last_checkpoint = Some(cp.clone());
            self.last_checkpoint_step = step;
            Some(cp)
        } else {
            None
        }
    }

    /// Recover from a checkpoint
    pub fn recover_from_checkpoint(
        checkpoint: ServerCheckpoint,
        config: ParamServerConfig,
    ) -> CoreResult<Self> {
        let server = ParameterServer::restore(&checkpoint, config)?;
        Ok(Self {
            server,
            last_checkpoint: Some(checkpoint),
            last_checkpoint_step: 0,
        })
    }

    /// Get the last checkpoint if available
    #[must_use]
    pub fn last_checkpoint(&self) -> Option<&ServerCheckpoint> {
        self.last_checkpoint.as_ref()
    }
}

/// Vector clock for causal ordering of distributed updates
///
/// Each worker maintains a logical clock. The vector clock tracks all
/// workers' clocks to determine causal relationships between events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    /// Logical clock value for each worker
    clocks: Vec<u64>,
}

impl VectorClock {
    /// Create a new vector clock for the given number of workers
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        Self {
            clocks: vec![0; num_workers],
        }
    }

    /// Increment the clock for a specific worker (local event)
    pub fn increment(&mut self, worker_id: usize) -> CoreResult<()> {
        if worker_id >= self.clocks.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Worker ID {worker_id} out of range (size = {})",
                self.clocks.len()
            ))));
        }
        self.clocks[worker_id] = self.clocks[worker_id].saturating_add(1);
        Ok(())
    }

    /// Merge with another vector clock (receive event)
    ///
    /// Takes the element-wise maximum of both clocks.
    pub fn merge(&mut self, other: &VectorClock) -> CoreResult<()> {
        if self.clocks.len() != other.clocks.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Vector clock size mismatch: {} vs {}",
                self.clocks.len(),
                other.clocks.len()
            ))));
        }
        for (mine, theirs) in self.clocks.iter_mut().zip(other.clocks.iter()) {
            *mine = (*mine).max(*theirs);
        }
        Ok(())
    }

    /// Check if this clock causally happened before `other`
    ///
    /// Returns true if all components of self are <= the corresponding
    /// components of other, and at least one is strictly less.
    #[must_use]
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        if self.clocks.len() != other.clocks.len() {
            return false;
        }
        let mut all_leq = true;
        let mut any_lt = false;
        for (a, b) in self.clocks.iter().zip(other.clocks.iter()) {
            if a > b {
                all_leq = false;
                break;
            }
            if a < b {
                any_lt = true;
            }
        }
        all_leq && any_lt
    }

    /// Check if two events are concurrent (neither happens-before the other)
    #[must_use]
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }

    /// Get the clock value for a specific worker
    pub fn get(&self, worker_id: usize) -> CoreResult<u64> {
        self.clocks.get(worker_id).copied().ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new(format!(
                "Worker ID {worker_id} out of range"
            )))
        })
    }

    /// Get the number of workers tracked
    #[must_use]
    pub fn len(&self) -> usize {
        self.clocks.len()
    }

    /// Check if the vector clock is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clocks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::param_server::types::{ConsistencyModel, ParameterUpdate};

    #[test]
    fn test_heartbeat_and_check() {
        let config = ParamServerConfig {
            num_workers: 3,
            ..ParamServerConfig::default()
        };
        let mut ftps = FaultTolerantPS::new(config);
        ftps.server_mut().register_worker();
        ftps.server_mut().register_worker();
        ftps.server_mut().register_worker();

        // All workers send heartbeats at t=10
        for i in 0..3 {
            ftps.heartbeat(i, 10).expect("heartbeat");
        }

        // At t=20 with timeout=15, no one is timed out
        let timed_out = ftps.check_workers(20, 15);
        assert!(timed_out.is_empty());

        // At t=30 with timeout=15, all timed out (last heartbeat was at 10)
        let timed_out = ftps.check_workers(30, 15);
        assert_eq!(timed_out.len(), 3);
    }

    #[test]
    fn test_handle_worker_failure() {
        let config = ParamServerConfig {
            num_workers: 2,
            consistency: ConsistencyModel::ASP,
            ..ParamServerConfig::default()
        };
        let mut ftps = FaultTolerantPS::new(config);
        ftps.server_mut().register_worker();
        ftps.server_mut().register_worker();

        ftps.handle_worker_failure(0).expect("mark dead");

        // Can't mark dead again
        assert!(ftps.handle_worker_failure(0).is_err());

        // Dead worker can't push
        let result = ftps.server_mut().push(ParameterUpdate {
            key: "k".into(),
            values: vec![1.0],
            worker_id: 0,
            version: 1,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_if_needed() {
        let config = ParamServerConfig {
            num_workers: 1,
            checkpoint_interval: 5,
            consistency: ConsistencyModel::ASP,
            ..ParamServerConfig::default()
        };
        let mut ftps = FaultTolerantPS::new(config);
        ftps.server_mut().register_worker();
        ftps.server_mut().init_parameter("p", vec![1.0]);

        // Step 3: no checkpoint
        assert!(ftps.checkpoint_if_needed(3).is_none());
        // Step 5: checkpoint
        assert!(ftps.checkpoint_if_needed(5).is_some());
        // Step 5 again: no double checkpoint
        assert!(ftps.checkpoint_if_needed(5).is_none());
        // Step 10: checkpoint
        assert!(ftps.checkpoint_if_needed(10).is_some());
    }

    #[test]
    fn test_recover_from_checkpoint() {
        let config = ParamServerConfig {
            num_workers: 1,
            consistency: ConsistencyModel::ASP,
            ..ParamServerConfig::default()
        };
        let mut ftps = FaultTolerantPS::new(config.clone());
        let w0 = ftps.server_mut().register_worker();
        ftps.server_mut().init_parameter("x", vec![99.0]);

        let cp = ftps.server().checkpoint();
        let recovered = FaultTolerantPS::recover_from_checkpoint(cp, config).expect("recover");
        let (vals, _) = recovered.server().pull("x", w0).expect("pull");
        assert!((vals[0] - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_clock_basic() {
        let mut vc1 = VectorClock::new(3);
        vc1.increment(0).expect("inc");
        vc1.increment(0).expect("inc");
        assert_eq!(vc1.get(0).expect("get"), 2);
        assert_eq!(vc1.get(1).expect("get"), 0);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let mut vc1 = VectorClock::new(2);
        vc1.increment(0).expect("inc");

        let mut vc2 = VectorClock::new(2);
        vc2.increment(0).expect("inc");
        vc2.increment(1).expect("inc");

        // vc1 = [1, 0], vc2 = [1, 1] => vc1 happens-before vc2
        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut vc1 = VectorClock::new(2);
        vc1.increment(0).expect("inc");
        // vc1 = [1, 0]

        let mut vc2 = VectorClock::new(2);
        vc2.increment(1).expect("inc");
        // vc2 = [0, 1]

        assert!(vc1.is_concurrent(&vc2));
        assert!(vc2.is_concurrent(&vc1));
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut vc1 = VectorClock::new(3);
        vc1.increment(0).expect("inc");
        vc1.increment(0).expect("inc");
        // vc1 = [2, 0, 0]

        let mut vc2 = VectorClock::new(3);
        vc2.increment(1).expect("inc");
        vc2.increment(2).expect("inc");
        vc2.increment(2).expect("inc");
        // vc2 = [0, 1, 2]

        vc1.merge(&vc2).expect("merge");
        // merged = [2, 1, 2]
        assert_eq!(vc1.get(0).expect("get"), 2);
        assert_eq!(vc1.get(1).expect("get"), 1);
        assert_eq!(vc1.get(2).expect("get"), 2);
    }

    #[test]
    fn test_vector_clock_size_mismatch() {
        let vc1 = VectorClock::new(2);
        let vc2 = VectorClock::new(3);
        assert!(!vc1.happens_before(&vc2));
    }

    // ── FaultTolerantPs (rolling-checkpoint + Instant heartbeat) tests ──

    fn make_ps_with_param() -> ParameterServer {
        use super::super::types::{AggregationMethod, ConsistencyModel};
        let config = ParamServerConfig {
            num_workers: 2,
            consistency: ConsistencyModel::ASP,
            aggregation: AggregationMethod::Mean,
            checkpoint_interval: 100,
            replication_factor: 1,
        };
        let mut ps = ParameterServer::new(config);
        ps.register_worker();
        ps.register_worker();
        ps.init_parameter("w", vec![1.0, 2.0, 3.0]);
        ps
    }

    #[test]
    fn test_fault_tolerant_ps_checkpoint_saved() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig {
            checkpoint_every: 10,
            max_checkpoints: 3,
        };
        let mut ftps = FaultTolerantPs::new(ps, cc);

        assert_eq!(ftps.n_checkpoints(), 0);
        ftps.checkpoint().expect("checkpoint");
        assert_eq!(ftps.n_checkpoints(), 1);
        ftps.checkpoint().expect("checkpoint 2");
        assert_eq!(ftps.n_checkpoints(), 2);
    }

    #[test]
    fn test_fault_tolerant_ps_rolling_eviction() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig {
            checkpoint_every: 5,
            max_checkpoints: 2,
        };
        let mut ftps = FaultTolerantPs::new(ps, cc);

        ftps.checkpoint().expect("cp1");
        ftps.checkpoint().expect("cp2");
        ftps.checkpoint().expect("cp3");
        // max_checkpoints = 2, so oldest is evicted
        assert_eq!(ftps.n_checkpoints(), 2);
    }

    #[test]
    fn test_fault_tolerant_ps_restore() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig::default();
        let mut ftps = FaultTolerantPs::new(ps, cc);

        ftps.checkpoint().expect("checkpoint before push");

        // Modify parameter
        use super::super::types::ParameterUpdate;
        ftps.server_mut()
            .push(ParameterUpdate {
                key: "w".into(),
                values: vec![99.0, 99.0, 99.0],
                worker_id: 0,
                version: 1,
            })
            .expect("push");

        // Restore to checkpoint
        ftps.restore_latest().expect("restore");

        let (vals, _) = ftps.server().pull("w", 0).expect("pull after restore");
        assert!((vals[0] - 1.0).abs() < f64::EPSILON);
        assert!((vals[1] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fault_tolerant_ps_restore_no_checkpoint() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig::default();
        let mut ftps = FaultTolerantPs::new(ps, cc);
        // No checkpoint taken yet
        assert!(ftps.restore_latest().is_err());
    }

    #[test]
    fn test_fault_tolerant_ps_heartbeat_no_failure() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig::default();
        let mut ftps = FaultTolerantPs::new(ps, cc);
        // Set a very long timeout so no failures are detected
        ftps.set_heartbeat_timeout_ms(60_000);
        ftps.heartbeat(0);
        ftps.heartbeat(1);
        let failed = ftps.detect_failed_workers();
        assert!(failed.is_empty(), "No workers should be failed: {failed:?}");
    }

    #[test]
    fn test_fault_tolerant_ps_detect_failed_workers() {
        let ps = make_ps_with_param();
        let cc = CheckpointConfig::default();
        let mut ftps = FaultTolerantPs::new(ps, cc);

        // Set a zero-ms timeout so workers are immediately considered failed
        ftps.set_heartbeat_timeout_ms(0);
        // Brief sleep to ensure at least 1 ms has elapsed
        std::thread::sleep(std::time::Duration::from_millis(2));

        let failed = ftps.detect_failed_workers();
        // Both workers should be detected as failed
        assert!(!failed.is_empty(), "Workers should be detected as failed");
    }

    #[test]
    fn test_checkpoint_config_default() {
        let cc = CheckpointConfig::default();
        assert_eq!(cc.checkpoint_every, 10);
        assert_eq!(cc.max_checkpoints, 3);
    }
}
