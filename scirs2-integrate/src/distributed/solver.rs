//! Distributed ODE solver implementation
//!
//! This module provides the main distributed ODE solver that coordinates
//! work distribution across compute nodes for large-scale integration problems.

use crate::common::IntegrateFloat;
use crate::distributed::checkpointing::{
    Checkpoint, CheckpointConfig, CheckpointGlobalState, CheckpointManager,
    FaultToleranceCoordinator, RecoveryAction,
};
use crate::distributed::communication::{BoundaryExchanger, Communicator, MessageChannel};
use crate::distributed::load_balancing::{ChunkDistributor, LoadBalancer, LoadBalancerConfig};
use crate::distributed::node::{ComputeNode, NodeManager};
use crate::distributed::types::{
    BoundaryData, ChunkId, ChunkResult, ChunkResultStatus, DistributedConfig, DistributedError,
    DistributedMetrics, DistributedResult, FaultToleranceMode, JobId, NodeId, NodeInfo, NodeStatus,
    WorkChunk,
};
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEMethod, ODEOptions};
use scirs2_core::ndarray::{array, Array1, ArrayView1};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Distributed ODE solver
pub struct DistributedODESolver<F: IntegrateFloat> {
    /// Node manager
    node_manager: Arc<NodeManager>,
    /// Load balancer
    load_balancer: Arc<LoadBalancer<F>>,
    /// Checkpoint manager
    checkpoint_manager: Arc<CheckpointManager<F>>,
    /// Fault tolerance coordinator
    fault_coordinator: Arc<FaultToleranceCoordinator<F>>,
    /// Message channels per node
    channels: RwLock<HashMap<NodeId, Arc<MessageChannel<F>>>>,
    /// Boundary exchanger
    boundary_exchanger: Arc<BoundaryExchanger<F>>,
    /// Configuration
    config: DistributedConfig<F>,
    /// Next job ID
    next_job_id: AtomicU64,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Active jobs
    active_jobs: RwLock<HashMap<JobId, JobState<F>>>,
    /// Metrics
    metrics: Mutex<DistributedMetrics>,
}

/// State of an active job
struct JobState<F: IntegrateFloat> {
    /// Job ID
    job_id: JobId,
    /// Time span
    t_span: (F, F),
    /// Initial state
    initial_state: Array1<F>,
    /// Total chunks
    total_chunks: usize,
    /// Completed chunks
    completed_chunks: Vec<ChunkResult<F>>,
    /// Pending chunks
    pending_chunks: Vec<ChunkId>,
    /// In-progress chunks
    in_progress_chunks: HashMap<ChunkId, NodeId>,
    /// Chunk ordering for assembly
    chunk_order: Vec<ChunkId>,
    /// Start time
    start_time: Instant,
    /// Last checkpoint time
    last_checkpoint: Option<Instant>,
    /// Chunks since last checkpoint
    chunks_since_checkpoint: usize,
}

impl<F: IntegrateFloat> DistributedODESolver<F> {
    /// Create a new distributed ODE solver
    pub fn new(config: DistributedConfig<F>) -> DistributedResult<Self> {
        let node_manager = Arc::new(NodeManager::new(config.heartbeat_interval));

        let load_balancer = Arc::new(LoadBalancer::new(
            config.load_balancing,
            LoadBalancerConfig::default(),
        ));

        let checkpoint_path = {
            let mut p = std::env::temp_dir();
            p.push("scirs_checkpoints");
            p
        };
        let checkpoint_config = CheckpointConfig {
            persist_to_disk: config.checkpointing_enabled,
            interval_chunks: config.checkpoint_interval,
            ..Default::default()
        };

        let checkpoint_manager =
            Arc::new(CheckpointManager::new(checkpoint_path, checkpoint_config)?);

        let fault_coordinator = Arc::new(FaultToleranceCoordinator::new(
            Arc::clone(&checkpoint_manager),
            config.fault_tolerance,
        ));

        let boundary_exchanger = Arc::new(BoundaryExchanger::new(config.communication_timeout));

        Ok(Self {
            node_manager,
            load_balancer,
            checkpoint_manager,
            fault_coordinator,
            channels: RwLock::new(HashMap::new()),
            boundary_exchanger,
            config,
            next_job_id: AtomicU64::new(1),
            shutdown: AtomicBool::new(false),
            active_jobs: RwLock::new(HashMap::new()),
            metrics: Mutex::new(DistributedMetrics::default()),
        })
    }

    /// Register a compute node
    pub fn register_node(&self, node: NodeInfo) -> DistributedResult<()> {
        let node_id = node.id;

        // Register with node manager
        self.node_manager
            .register_node(node.address, node.capabilities.clone())?;

        // Register with load balancer
        self.load_balancer.register_node(node_id)?;

        // Create message channel
        let channel = Arc::new(MessageChannel::new(self.config.communication_timeout));
        if let Ok(mut channels) = self.channels.write() {
            channels.insert(node_id, channel);
        }

        Ok(())
    }

    /// Deregister a compute node
    pub fn deregister_node(&self, node_id: NodeId) -> DistributedResult<()> {
        self.node_manager.deregister_node(node_id)?;
        self.load_balancer.deregister_node(node_id)?;

        if let Ok(mut channels) = self.channels.write() {
            channels.remove(&node_id);
        }

        Ok(())
    }

    /// Solve an ODE problem distributedly
    pub fn solve<Func>(
        &self,
        f: Func,
        t_span: (F, F),
        y0: Array1<F>,
        options: Option<ODEOptions<F>>,
    ) -> IntegrateResult<DistributedODEResult<F>>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Send + Sync + Clone + 'static,
    {
        let start_time = Instant::now();

        // Get available nodes
        let available_nodes = self.node_manager.get_available_nodes();
        if available_nodes.is_empty() {
            return Err(IntegrateError::ComputationError(
                "No compute nodes available".to_string(),
            ));
        }

        // Create job
        let job_id = JobId::new(self.next_job_id.fetch_add(1, Ordering::SeqCst));

        // Calculate number of chunks based on nodes
        let num_chunks = (available_nodes.len() * self.config.chunks_per_node).max(1);

        // Create chunk distributor and generate chunks
        let distributor = ChunkDistributor::new(job_id);
        let chunks = distributor.create_chunks(t_span, y0.clone(), num_chunks);

        // Initialize job state
        let chunk_order: Vec<ChunkId> = chunks.iter().map(|c| c.id).collect();
        let pending_chunks = chunk_order.clone();

        let job_state = JobState {
            job_id,
            t_span,
            initial_state: y0.clone(),
            total_chunks: num_chunks,
            completed_chunks: Vec::new(),
            pending_chunks,
            in_progress_chunks: HashMap::new(),
            chunk_order,
            start_time,
            last_checkpoint: None,
            chunks_since_checkpoint: 0,
        };

        // Register job
        if let Ok(mut jobs) = self.active_jobs.write() {
            jobs.insert(job_id, job_state);
        }

        // Distribute initial work
        self.distribute_chunks(job_id, chunks, &available_nodes, &f)?;

        // Wait for completion
        let result = self.wait_for_completion(job_id, &f)?;

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_processing_time += start_time.elapsed();
        }

        // Cleanup job
        if let Ok(mut jobs) = self.active_jobs.write() {
            jobs.remove(&job_id);
        }

        Ok(result)
    }

    /// Distribute chunks to nodes
    fn distribute_chunks<Func>(
        &self,
        job_id: JobId,
        chunks: Vec<WorkChunk<F>>,
        nodes: &[NodeInfo],
        f: &Func,
    ) -> IntegrateResult<()>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Send + Sync + Clone + 'static,
    {
        for chunk in chunks {
            let node_id = self
                .load_balancer
                .assign_chunk(&chunk, nodes)
                .map_err(|e| IntegrateError::ComputationError(e.to_string()))?;

            // Record assignment
            if let Ok(mut jobs) = self.active_jobs.write() {
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.pending_chunks.retain(|id| *id != chunk.id);
                    job.in_progress_chunks.insert(chunk.id, node_id);
                }
            }

            // In a real implementation, this would send the chunk over the network
            // For now, we simulate local processing
        }

        Ok(())
    }

    /// Wait for job completion
    fn wait_for_completion<Func>(
        &self,
        job_id: JobId,
        f: &Func,
    ) -> IntegrateResult<DistributedODEResult<F>>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Send + Sync + Clone + 'static,
    {
        let timeout = Duration::from_secs(3600); // 1 hour timeout
        let deadline = Instant::now() + timeout;

        loop {
            if Instant::now() > deadline {
                return Err(IntegrateError::ConvergenceError(
                    "Distributed solve timeout".to_string(),
                ));
            }

            // Check completion
            let (is_complete, needs_processing) = {
                let jobs = self.active_jobs.read().map_err(|_| {
                    IntegrateError::ComputationError("Failed to read job state".to_string())
                })?;

                if let Some(job) = jobs.get(&job_id) {
                    let complete =
                        job.pending_chunks.is_empty() && job.in_progress_chunks.is_empty();
                    let needs = !job.in_progress_chunks.is_empty();
                    (complete, needs)
                } else {
                    return Err(IntegrateError::ComputationError(
                        "Job not found".to_string(),
                    ));
                }
            };

            if is_complete {
                break;
            }

            if needs_processing {
                // Simulate processing chunks
                self.process_pending_chunks(job_id, f)?;
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        // Assemble result
        self.assemble_result(job_id)
    }

    /// Process pending chunks (simulation for local testing)
    ///
    /// Chunks are processed sequentially in chunk_order so that each chunk
    /// can use the final state from the previous chunk as its initial state.
    fn process_pending_chunks<Func>(&self, job_id: JobId, f: &Func) -> IntegrateResult<()>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Send + Sync + Clone + 'static,
    {
        // Get ordered list of in-progress chunk IDs and their assigned nodes
        let ordered_chunks: Vec<(ChunkId, NodeId, usize)> = {
            let jobs = self.active_jobs.read().map_err(|_| {
                IntegrateError::ComputationError("Failed to read job state".to_string())
            })?;

            if let Some(job) = jobs.get(&job_id) {
                let mut items: Vec<(ChunkId, NodeId, usize)> = job
                    .in_progress_chunks
                    .iter()
                    .map(|(chunk_id, node_id)| {
                        let idx = job
                            .chunk_order
                            .iter()
                            .position(|id| id == chunk_id)
                            .unwrap_or(0);
                        (*chunk_id, *node_id, idx)
                    })
                    .collect();
                // Sort by chunk order index so we process them sequentially
                items.sort_by_key(|&(_, _, idx)| idx);
                items
            } else {
                Vec::new()
            }
        };

        // Process each chunk one at a time, in order
        for (chunk_id, node_id, idx) in ordered_chunks {
            // Build the work chunk with correct initial state from completed chunks
            let chunk = {
                let jobs = self.active_jobs.read().map_err(|_| {
                    IntegrateError::ComputationError("Failed to read job state".to_string())
                })?;
                let job = jobs
                    .get(&job_id)
                    .ok_or_else(|| IntegrateError::ComputationError("Job not found".to_string()))?;

                let (t_start, t_end) = job.t_span;
                let dt = (t_end - t_start) / F::from(job.total_chunks).unwrap_or(F::one());

                let chunk_t_start = t_start + dt * F::from(idx).unwrap_or(F::zero());
                let chunk_t_end = if idx == job.total_chunks - 1 {
                    t_end
                } else {
                    t_start + dt * F::from(idx + 1).unwrap_or(F::one())
                };

                // Get initial state from previous chunk result or job initial state
                let initial_state = if idx == 0 {
                    job.initial_state.clone()
                } else {
                    let prev_chunk_id = job.chunk_order.get(idx - 1).ok_or_else(|| {
                        IntegrateError::ComputationError(
                            "Previous chunk not found in order".to_string(),
                        )
                    })?;
                    job.completed_chunks
                        .iter()
                        .find(|r| r.chunk_id == *prev_chunk_id)
                        .map(|r| r.final_state.clone())
                        .unwrap_or_else(|| job.initial_state.clone())
                };

                WorkChunk::new(
                    chunk_id,
                    job_id,
                    (chunk_t_start, chunk_t_end),
                    initial_state,
                )
            };

            let result = self.process_single_chunk(&chunk, node_id, f)?;

            // Update job state
            if let Ok(mut jobs) = self.active_jobs.write() {
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.in_progress_chunks.remove(&chunk_id);
                    job.completed_chunks.push(result);
                    job.chunks_since_checkpoint += 1;

                    // Check if checkpoint is needed
                    if self.config.checkpointing_enabled
                        && self
                            .checkpoint_manager
                            .should_checkpoint(job.chunks_since_checkpoint)
                    {
                        let global_state = CheckpointGlobalState {
                            iteration: 0,
                            chunks_completed: job.completed_chunks.len(),
                            chunks_remaining: job.pending_chunks.len()
                                + job.in_progress_chunks.len(),
                            current_time: F::zero(),
                            error_estimate: F::zero(),
                        };

                        let _ = self.checkpoint_manager.create_checkpoint(
                            job_id,
                            job.completed_chunks.clone(),
                            job.in_progress_chunks.keys().cloned().collect(),
                            global_state,
                        );

                        job.chunks_since_checkpoint = 0;
                        job.last_checkpoint = Some(Instant::now());
                    }
                }
            }

            // Update load balancer
            let processing_time = Duration::from_millis(10); // Simulated
            self.load_balancer.report_completion(
                node_id,
                chunk.estimated_cost,
                processing_time,
                true,
            );
        }

        Ok(())
    }

    /// Process a single chunk using local ODE solver
    fn process_single_chunk<Func>(
        &self,
        chunk: &WorkChunk<F>,
        node_id: NodeId,
        f: &Func,
    ) -> IntegrateResult<ChunkResult<F>>
    where
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Send + Sync + Clone + 'static,
    {
        let start_time = Instant::now();

        // Use RK4 for simplicity
        let (t_start, t_end) = chunk.time_interval;
        let mut t = t_start;
        let mut y = chunk.initial_state.clone();

        let n_steps = 100;
        let h = (t_end - t_start) / F::from(n_steps).unwrap_or(F::one());

        let mut time_points = vec![t_start];
        let mut states = vec![y.clone()];

        for _ in 0..n_steps {
            // RK4 step
            let k1 = f(t, y.view());
            let k2 = f(
                t + h / F::from(2.0).unwrap_or(F::one()),
                (&y + &(&k1 * h / F::from(2.0).unwrap_or(F::one()))).view(),
            );
            let k3 = f(
                t + h / F::from(2.0).unwrap_or(F::one()),
                (&y + &(&k2 * h / F::from(2.0).unwrap_or(F::one()))).view(),
            );
            let k4 = f(t + h, (&y + &(&k3 * h)).view());

            y = &y
                + &((&k1
                    + &(&k2 * F::from(2.0).unwrap_or(F::one()))
                    + &(&k3 * F::from(2.0).unwrap_or(F::one()))
                    + &k4)
                    * h
                    / F::from(6.0).unwrap_or(F::one()));
            t += h;

            time_points.push(t);
            states.push(y.clone());
        }

        let final_state = y.clone();
        let final_derivative = Some(f(t, y.view()));

        Ok(ChunkResult {
            chunk_id: chunk.id,
            node_id,
            time_points,
            states,
            final_state,
            final_derivative,
            error_estimate: F::from(1e-6).unwrap_or(F::epsilon()),
            processing_time: start_time.elapsed(),
            memory_used: 0,
            status: ChunkResultStatus::Success,
        })
    }

    /// Assemble final result from completed chunks
    fn assemble_result(&self, job_id: JobId) -> IntegrateResult<DistributedODEResult<F>> {
        let jobs = self.active_jobs.read().map_err(|_| {
            IntegrateError::ComputationError("Failed to read job state".to_string())
        })?;

        let job = jobs
            .get(&job_id)
            .ok_or_else(|| IntegrateError::ComputationError("Job not found".to_string()))?;

        // Sort results by chunk order
        let mut sorted_results: Vec<_> = job.completed_chunks.clone();
        sorted_results.sort_by_key(|r| {
            job.chunk_order
                .iter()
                .position(|id| *id == r.chunk_id)
                .unwrap_or(usize::MAX)
        });

        // Concatenate time points and states
        let mut t_all = Vec::new();
        let mut y_all = Vec::new();

        for (i, result) in sorted_results.iter().enumerate() {
            let skip_first = if i > 0 { 1 } else { 0 };
            t_all.extend(result.time_points.iter().skip(skip_first).cloned());
            y_all.extend(result.states.iter().skip(skip_first).cloned());
        }

        let total_time = job.start_time.elapsed();

        // Get metrics
        let metrics = self.metrics.lock().map(|m| m.clone()).unwrap_or_default();

        Ok(DistributedODEResult {
            t: t_all,
            y: y_all,
            job_id,
            chunks_processed: job.completed_chunks.len(),
            nodes_used: job
                .completed_chunks
                .iter()
                .map(|r| r.node_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            total_time,
            metrics,
        })
    }

    /// Shutdown the solver
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.node_manager.stop_health_monitoring();
    }

    /// Get solver metrics
    pub fn get_metrics(&self) -> DistributedMetrics {
        self.metrics.lock().map(|m| m.clone()).unwrap_or_default()
    }
}

/// Result of a distributed ODE solve
#[derive(Debug, Clone)]
pub struct DistributedODEResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Solution states
    pub y: Vec<Array1<F>>,
    /// Job ID
    pub job_id: JobId,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Number of nodes used
    pub nodes_used: usize,
    /// Total time taken
    pub total_time: Duration,
    /// Distributed metrics
    pub metrics: DistributedMetrics,
}

impl<F: IntegrateFloat> DistributedODEResult<F> {
    /// Get final state
    pub fn final_state(&self) -> Option<&Array1<F>> {
        self.y.last()
    }

    /// Get state at specific index
    pub fn state_at(&self, index: usize) -> Option<&Array1<F>> {
        self.y.get(index)
    }

    /// Get number of time points
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Interpolate solution at a given time
    pub fn interpolate(&self, t_target: F) -> Option<Array1<F>> {
        if self.t.is_empty() {
            return None;
        }

        // Find bracketing points
        let mut left_idx = 0;
        for (i, &t) in self.t.iter().enumerate() {
            if t <= t_target {
                left_idx = i;
            } else {
                break;
            }
        }

        let right_idx = (left_idx + 1).min(self.t.len() - 1);

        if left_idx == right_idx {
            return self.y.get(left_idx).cloned();
        }

        // Linear interpolation
        let t_left = self.t[left_idx];
        let t_right = self.t[right_idx];
        let dt = t_right - t_left;

        if dt.abs() < F::epsilon() {
            return self.y.get(left_idx).cloned();
        }

        let alpha = (t_target - t_left) / dt;
        let y_left = &self.y[left_idx];
        let y_right = &self.y[right_idx];

        Some(y_left * (F::one() - alpha) + y_right * alpha)
    }
}

/// Builder for distributed ODE solver
pub struct DistributedODESolverBuilder<F: IntegrateFloat> {
    config: DistributedConfig<F>,
}

impl<F: IntegrateFloat> DistributedODESolverBuilder<F> {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: DistributedConfig::default(),
        }
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: F) -> Self {
        self.config.tolerance = tol;
        self
    }

    /// Set chunks per node
    pub fn chunks_per_node(mut self, n: usize) -> Self {
        self.config.chunks_per_node = n;
        self
    }

    /// Enable checkpointing
    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.config.checkpointing_enabled = true;
        self.config.checkpoint_interval = interval;
        self
    }

    /// Set fault tolerance mode
    pub fn fault_tolerance(mut self, mode: FaultToleranceMode) -> Self {
        self.config.fault_tolerance = mode;
        self
    }

    /// Set communication timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.communication_timeout = timeout;
        self
    }

    /// Build the solver
    pub fn build(self) -> DistributedResult<DistributedODESolver<F>> {
        DistributedODESolver::new(self.config)
    }
}

impl<F: IntegrateFloat> Default for DistributedODESolverBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::types::NodeCapabilities;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn create_test_node(id: u64) -> NodeInfo {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080 + id as u16);
        let mut info = NodeInfo::new(NodeId::new(id), addr);
        info.capabilities = NodeCapabilities::default();
        info.status = NodeStatus::Available;
        info
    }

    #[test]
    fn test_distributed_solver_creation() {
        let config = DistributedConfig::<f64>::default();
        let solver = DistributedODESolver::new(config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_distributed_solver_node_registration() {
        let config = DistributedConfig::<f64>::default();
        let solver = DistributedODESolver::new(config).expect("Failed to create solver");

        let node = create_test_node(1);
        let result = solver.register_node(node);
        assert!(result.is_ok());
    }

    #[test]
    fn test_distributed_solve_simple_ode() {
        let config = DistributedConfig::<f64>::default();
        let solver = DistributedODESolver::new(config).expect("Failed to create solver");

        // Register test nodes
        for i in 0..2 {
            let node = create_test_node(i);
            solver.register_node(node).expect("Failed to register node");
        }

        // Solve y' = -y, y(0) = 1
        let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];
        let y0 = array![1.0];

        let result = solver.solve(f, (0.0, 1.0), y0, None);
        assert!(result.is_ok());

        let result = result.expect("Solve failed");
        assert!(!result.t.is_empty());
        assert!(!result.y.is_empty());

        // Final value should be close to e^(-1)
        let expected = (-1.0_f64).exp();
        let actual = result.final_state().expect("No final state")[0];
        assert!((actual - expected).abs() < 0.01);
    }

    #[test]
    fn test_distributed_result_interpolation() {
        let result = DistributedODEResult::<f64> {
            t: vec![0.0, 0.5, 1.0],
            y: vec![array![1.0], array![0.6], array![0.4]],
            job_id: JobId::new(1),
            chunks_processed: 1,
            nodes_used: 1,
            total_time: Duration::from_secs(1),
            metrics: DistributedMetrics::default(),
        };

        let interpolated = result.interpolate(0.25).expect("Interpolation failed");
        assert!((interpolated[0] - 0.8_f64).abs() < 0.01_f64);
    }

    #[test]
    fn test_solver_builder() {
        let solver = DistributedODESolverBuilder::<f64>::new()
            .tolerance(1e-8)
            .chunks_per_node(8)
            .with_checkpointing(5)
            .fault_tolerance(FaultToleranceMode::Standard)
            .timeout(Duration::from_secs(60))
            .build();

        assert!(solver.is_ok());
    }
}
