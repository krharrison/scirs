//! Parameter server with multiple consistency models
//!
//! Supports BSP (Bulk Synchronous Parallel), ASP (Asynchronous Parallel),
//! and SSP (Stale Synchronous Parallel) consistency models.

use std::collections::HashMap;

use crate::error::{CoreError, CoreResult, ErrorContext};

use super::types::{
    AggregationMethod, ConsistencyModel, ParamServerConfig, ParameterUpdate, WorkerState,
};

/// A checkpoint of the parameter server state
#[derive(Debug, Clone)]
pub struct ServerCheckpoint {
    /// Snapshot of all parameters and their versions
    pub parameters: HashMap<String, (Vec<f64>, u64)>,
    /// Global version at checkpoint time
    pub version: u64,
    /// Snapshot of all worker states
    pub worker_states: Vec<WorkerState>,
}

/// Parameter server supporting BSP, ASP, and SSP consistency models
#[derive(Debug)]
pub struct ParameterServer {
    /// Configuration
    config: ParamServerConfig,
    /// In-memory parameter store: key -> (values, version)
    parameters: HashMap<String, (Vec<f64>, u64)>,
    /// Registered workers
    workers: Vec<WorkerState>,
    /// Global version counter
    global_version: u64,
    /// Buffered updates for BSP/SSP (key -> list of updates)
    update_buffer: HashMap<String, Vec<ParameterUpdate>>,
    /// Set of worker IDs that have pushed in the current BSP round
    bsp_pushed_workers: Vec<bool>,
}

impl ParameterServer {
    /// Create a new parameter server with the given configuration
    #[must_use]
    pub fn new(config: ParamServerConfig) -> Self {
        let num_workers = config.num_workers;
        Self {
            config,
            parameters: HashMap::new(),
            workers: Vec::new(),
            global_version: 0,
            update_buffer: HashMap::new(),
            bsp_pushed_workers: vec![false; num_workers],
        }
    }

    /// Register a new worker and return its ID
    pub fn register_worker(&mut self) -> usize {
        let worker_id = self.workers.len();
        self.workers.push(WorkerState::new(worker_id));
        // Extend BSP tracking if needed
        if self.bsp_pushed_workers.len() <= worker_id {
            self.bsp_pushed_workers.resize(worker_id + 1, false);
        }
        worker_id
    }

    /// Initialize a parameter with the given key and values
    pub fn init_parameter(&mut self, key: impl Into<String>, values: Vec<f64>) {
        let key = key.into();
        self.parameters.entry(key).or_insert((values, 0));
    }

    /// Push an update from a worker
    ///
    /// Behavior depends on the consistency model:
    /// - BSP: buffers until all workers push, then applies via `barrier_sync`
    /// - ASP: applies immediately
    /// - SSP: applies if within staleness bound, otherwise buffers
    pub fn push(&mut self, update: ParameterUpdate) -> CoreResult<()> {
        let worker_id = update.worker_id;
        if worker_id >= self.workers.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Unknown worker ID: {worker_id}"
            ))));
        }
        if !self.workers[worker_id].is_alive {
            return Err(CoreError::ComputationError(ErrorContext::new(format!(
                "Worker {worker_id} is not alive"
            ))));
        }

        // Update worker version
        self.workers[worker_id].version = update.version;

        match &self.config.consistency {
            ConsistencyModel::BSP => {
                // Buffer the update
                self.update_buffer
                    .entry(update.key.clone())
                    .or_default()
                    .push(update);
                self.bsp_pushed_workers[worker_id] = true;
            }
            ConsistencyModel::ASP => {
                // Apply immediately
                self.apply_single_update(&update)?;
            }
            ConsistencyModel::SSP { max_staleness } => {
                let min_version = self
                    .workers
                    .iter()
                    .filter(|w| w.is_alive)
                    .map(|w| w.version)
                    .min()
                    .unwrap_or(0);
                let staleness = update.version.saturating_sub(min_version) as usize;

                if staleness <= *max_staleness {
                    // Within staleness bound — apply immediately
                    self.apply_single_update(&update)?;
                } else {
                    // Too stale — buffer until slower workers catch up
                    self.update_buffer
                        .entry(update.key.clone())
                        .or_default()
                        .push(update);
                }
            }
        }
        Ok(())
    }

    /// Pull current parameter values for a given key
    pub fn pull(&self, key: &str, worker_id: usize) -> CoreResult<(Vec<f64>, u64)> {
        if worker_id >= self.workers.len() {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Unknown worker ID: {worker_id}"
            ))));
        }
        self.parameters
            .get(key)
            .cloned()
            .ok_or_else(|| CoreError::ValueError(ErrorContext::new(format!("Unknown key: {key}"))))
    }

    /// Aggregate updates using the configured method
    #[must_use]
    pub fn aggregate_updates(updates: &[ParameterUpdate], method: &AggregationMethod) -> Vec<f64> {
        if updates.is_empty() {
            return Vec::new();
        }
        let dim = updates[0].values.len();

        match method {
            AggregationMethod::Mean => {
                let mut sum = vec![0.0; dim];
                for u in updates {
                    for (s, v) in sum.iter_mut().zip(u.values.iter()) {
                        *s += v;
                    }
                }
                let n = updates.len() as f64;
                sum.iter().map(|s| s / n).collect()
            }
            AggregationMethod::Sum => {
                let mut sum = vec![0.0; dim];
                for u in updates {
                    for (s, v) in sum.iter_mut().zip(u.values.iter()) {
                        *s += v;
                    }
                }
                sum
            }
            AggregationMethod::WeightedMean { weights } => {
                let mut weighted_sum = vec![0.0; dim];
                let mut total_weight = 0.0;
                for u in updates {
                    let w = weights.get(u.worker_id).copied().unwrap_or(1.0);
                    total_weight += w;
                    for (s, v) in weighted_sum.iter_mut().zip(u.values.iter()) {
                        *s += v * w;
                    }
                }
                if total_weight.abs() < f64::EPSILON {
                    weighted_sum
                } else {
                    weighted_sum.iter().map(|s| s / total_weight).collect()
                }
            }
        }
    }

    /// BSP barrier synchronization
    ///
    /// Applies all buffered updates and increments the global version.
    /// Returns an error if not all alive workers have pushed.
    pub fn barrier_sync(&mut self) -> CoreResult<()> {
        // Check that all alive workers have pushed
        for w in &self.workers {
            if w.is_alive
                && !self
                    .bsp_pushed_workers
                    .get(w.worker_id)
                    .copied()
                    .unwrap_or(false)
            {
                return Err(CoreError::ComputationError(ErrorContext::new(format!(
                    "BSP barrier: worker {} has not pushed yet",
                    w.worker_id
                ))));
            }
        }

        // Apply all buffered updates
        let keys: Vec<String> = self.update_buffer.keys().cloned().collect();
        for key in &keys {
            if let Some(updates) = self.update_buffer.get(key) {
                let aggregated = Self::aggregate_updates(updates, &self.config.aggregation);
                if !aggregated.is_empty() {
                    let version = self.global_version + 1;
                    self.parameters.insert(key.clone(), (aggregated, version));
                }
            }
        }

        // Clear buffer and reset BSP state
        self.update_buffer.clear();
        for flag in &mut self.bsp_pushed_workers {
            *flag = false;
        }
        self.global_version += 1;

        Ok(())
    }

    /// Create a checkpoint of the current server state
    #[must_use]
    pub fn checkpoint(&self) -> ServerCheckpoint {
        ServerCheckpoint {
            parameters: self.parameters.clone(),
            version: self.global_version,
            worker_states: self.workers.clone(),
        }
    }

    /// Restore a parameter server from a checkpoint
    pub fn restore(checkpoint: &ServerCheckpoint, config: ParamServerConfig) -> CoreResult<Self> {
        let num_workers = config.num_workers;
        Ok(Self {
            config,
            parameters: checkpoint.parameters.clone(),
            workers: checkpoint.worker_states.clone(),
            global_version: checkpoint.version,
            update_buffer: HashMap::new(),
            bsp_pushed_workers: vec![false; num_workers],
        })
    }

    /// Get the current global version
    #[must_use]
    pub fn global_version(&self) -> u64 {
        self.global_version
    }

    /// Get number of registered workers
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get a reference to the worker states
    #[must_use]
    pub fn workers(&self) -> &[WorkerState] {
        &self.workers
    }

    /// Get a mutable reference to the worker states
    pub fn workers_mut(&mut self) -> &mut Vec<WorkerState> {
        &mut self.workers
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &ParamServerConfig {
        &self.config
    }

    /// Apply a single update directly to the parameter store
    fn apply_single_update(&mut self, update: &ParameterUpdate) -> CoreResult<()> {
        let entry = self
            .parameters
            .entry(update.key.clone())
            .or_insert_with(|| (vec![0.0; update.values.len()], 0));

        // For single-worker updates, just replace values and bump version
        entry.0 = update.values.clone();
        entry.1 = update.version;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_workers() {
        let config = ParamServerConfig::default();
        let mut ps = ParameterServer::new(config);
        let id0 = ps.register_worker();
        let id1 = ps.register_worker();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(ps.num_workers(), 2);
    }

    #[test]
    fn test_init_and_pull() {
        let config = ParamServerConfig::default();
        let mut ps = ParameterServer::new(config);
        let wid = ps.register_worker();
        ps.init_parameter("w1", vec![1.0, 2.0, 3.0]);
        let (vals, ver) = ps.pull("w1", wid).expect("pull should succeed");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        assert_eq!(ver, 0);
    }

    #[test]
    fn test_pull_unknown_key() {
        let config = ParamServerConfig::default();
        let mut ps = ParameterServer::new(config);
        let wid = ps.register_worker();
        let result = ps.pull("nonexistent", wid);
        assert!(result.is_err());
    }

    #[test]
    fn test_bsp_push_and_barrier() {
        let config = ParamServerConfig {
            num_workers: 2,
            consistency: ConsistencyModel::BSP,
            aggregation: AggregationMethod::Mean,
            ..ParamServerConfig::default()
        };
        let mut ps = ParameterServer::new(config);
        let w0 = ps.register_worker();
        let w1 = ps.register_worker();
        ps.init_parameter("w", vec![0.0, 0.0]);

        // Worker 0 pushes
        ps.push(ParameterUpdate {
            key: "w".to_string(),
            values: vec![2.0, 4.0],
            worker_id: w0,
            version: 1,
        })
        .expect("push w0");

        // Barrier should fail (w1 hasn't pushed)
        assert!(ps.barrier_sync().is_err());

        // Worker 1 pushes
        ps.push(ParameterUpdate {
            key: "w".to_string(),
            values: vec![4.0, 6.0],
            worker_id: w1,
            version: 1,
        })
        .expect("push w1");

        // Barrier should succeed now
        ps.barrier_sync().expect("barrier");

        let (vals, ver) = ps.pull("w", w0).expect("pull after barrier");
        // Mean of [2,4] and [4,6] = [3,5]
        assert!((vals[0] - 3.0).abs() < f64::EPSILON);
        assert!((vals[1] - 5.0).abs() < f64::EPSILON);
        assert_eq!(ver, 1);
    }

    #[test]
    fn test_asp_push() {
        let config = ParamServerConfig {
            num_workers: 1,
            consistency: ConsistencyModel::ASP,
            ..ParamServerConfig::default()
        };
        let mut ps = ParameterServer::new(config);
        let w0 = ps.register_worker();
        ps.init_parameter("p", vec![0.0]);

        ps.push(ParameterUpdate {
            key: "p".to_string(),
            values: vec![42.0],
            worker_id: w0,
            version: 1,
        })
        .expect("asp push");

        let (vals, _) = ps.pull("p", w0).expect("pull");
        assert!((vals[0] - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ssp_within_bound() {
        let config = ParamServerConfig {
            num_workers: 2,
            consistency: ConsistencyModel::SSP { max_staleness: 2 },
            ..ParamServerConfig::default()
        };
        let mut ps = ParameterServer::new(config);
        let w0 = ps.register_worker();
        let _w1 = ps.register_worker();

        ps.push(ParameterUpdate {
            key: "s".to_string(),
            values: vec![10.0],
            worker_id: w0,
            version: 1,
        })
        .expect("ssp push within bound");

        // Should be applied immediately since staleness = 1 - 0 = 1 <= 2
        let (vals, _) = ps.pull("s", w0).expect("pull");
        assert!((vals[0] - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_sum() {
        let updates = vec![
            ParameterUpdate {
                key: "k".into(),
                values: vec![1.0, 2.0],
                worker_id: 0,
                version: 1,
            },
            ParameterUpdate {
                key: "k".into(),
                values: vec![3.0, 4.0],
                worker_id: 1,
                version: 1,
            },
        ];
        let result = ParameterServer::aggregate_updates(&updates, &AggregationMethod::Sum);
        assert!((result[0] - 4.0).abs() < f64::EPSILON);
        assert!((result[1] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_checkpoint_and_restore() {
        let config = ParamServerConfig {
            num_workers: 1,
            consistency: ConsistencyModel::ASP,
            ..ParamServerConfig::default()
        };
        let mut ps = ParameterServer::new(config.clone());
        let w0 = ps.register_worker();
        ps.init_parameter("x", vec![1.0, 2.0]);
        ps.push(ParameterUpdate {
            key: "x".to_string(),
            values: vec![5.0, 6.0],
            worker_id: w0,
            version: 1,
        })
        .expect("push");

        let cp = ps.checkpoint();
        let restored = ParameterServer::restore(&cp, config).expect("restore");
        let (vals, _) = restored.pull("x", w0).expect("pull from restored");
        assert!((vals[0] - 5.0).abs() < f64::EPSILON);
        assert!((vals[1] - 6.0).abs() < f64::EPSILON);
    }
}
