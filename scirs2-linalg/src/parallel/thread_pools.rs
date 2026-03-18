//! Enhanced thread pool configurations and management
//!
//! This module provides sophisticated thread pool management with support for
//! adaptive sizing, CPU affinity, NUMA awareness, and workload-specific optimization.

use crate::error::LinalgResult;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Thread pool profile for different types of workloads
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThreadPoolProfile {
    /// Optimized for CPU-intensive computations
    CpuIntensive,
    /// Optimized for memory-bound operations
    MemoryBound,
    /// Balanced profile for mixed workloads
    Balanced,
    /// Low-latency profile for quick operations
    LowLatency,
    /// High-throughput profile for bulk operations
    HighThroughput,
    /// Linear algebra specific optimizations
    LinearAlgebra,
    /// Matrix multiplication optimized
    MatrixMultiplication,
    /// Eigenvalue computation optimized
    EigenComputation,
    /// Decomposition algorithms optimized
    Decomposition,
    /// Iterative solver optimized
    IterativeSolver,
    /// NUMA-aware parallel processing
    NumaOptimized,
    /// GPU-CPU hybrid processing
    HybridComputing,
    /// Custom profile with specific parameters
    Custom(String),
}

/// Thread affinity strategy
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityStrategy {
    /// No specific affinity
    None,
    /// Pin threads to specific cores
    Pinned(Vec<usize>),
    /// Spread threads across NUMA nodes
    NumaSpread,
    /// Keep threads within same NUMA node
    NumaCompact,
    /// Custom affinity pattern
    Custom(Vec<Option<usize>>),
}

/// Thread pool configuration
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadPoolConfig {
    /// Profile for workload optimization
    pub profile: ThreadPoolProfile,
    /// Minimum number of threads
    pub min_threads: usize,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Current number of active threads
    pub active_threads: usize,
    /// Thread idle timeout
    pub idle_timeout: Duration,
    /// Affinity strategy
    pub affinity: AffinityStrategy,
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// Work stealing enabled
    pub work_stealing: bool,
    /// Queue capacity
    pub queue_capacity: usize,
    /// Thread stack size
    pub stacksize: Option<usize>,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::Balanced,
            min_threads: 1,
            max_threads: num_cpus,
            active_threads: num_cpus,
            idle_timeout: Duration::from_secs(60),
            affinity: AffinityStrategy::None,
            numa_aware: false,
            work_stealing: true,
            queue_capacity: 1024,
            stacksize: None,
        }
    }
}

/// Advanced thread pool configuration with monitoring and adaptation
#[derive(Debug, Clone)]
pub struct AdvancedThreadPoolConfig {
    /// Base configuration
    pub base_config: ThreadPoolConfig,
    /// Dynamic sizing configuration
    pub dynamic_sizing: DynamicSizingConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Resource isolation configuration
    pub resource_isolation: ResourceIsolationConfig,
    /// Workload adaptation configuration
    pub workload_adaptation: WorkloadAdaptationConfig,
}

/// Dynamic thread pool sizing configuration
#[derive(Debug, Clone)]
pub struct DynamicSizingConfig {
    /// Enable dynamic resizing
    pub enabled: bool,
    /// CPU utilization threshold for scaling up
    pub scale_up_threshold: f64,
    /// CPU utilization threshold for scaling down
    pub scale_down_threshold: f64,
    /// Minimum observation period before scaling decisions
    pub observation_period: Duration,
    /// Maximum scaling factor per adjustment
    pub max_scaling_factor: f64,
}

impl Default for DynamicSizingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            observation_period: Duration::from_secs(5),
            max_scaling_factor: 1.5,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Maximum history entries
    pub max_history_entries: usize,
    /// Enable detailed CPU metrics
    pub detailed_cpu_metrics: bool,
    /// Enable memory usage tracking
    pub memory_tracking: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(1),
            max_history_entries: 1000,
            detailed_cpu_metrics: false,
            memory_tracking: true,
        }
    }
}

/// Resource isolation configuration
#[derive(Debug, Clone)]
pub struct ResourceIsolationConfig {
    /// Memory allocation policy
    pub memory_policy: CacheAllocationPolicy,
    /// CPU core isolation
    pub cpu_isolation: bool,
    /// Memory bandwidth limits
    pub memory_bandwidth_limit: Option<f64>,
    /// Cache partition assignment
    pub cache_partition: Option<usize>,
}

/// Cache allocation policy
#[derive(Debug, Clone)]
pub enum CacheAllocationPolicy {
    /// Default system policy
    Default,
    /// Shared cache allocation
    Shared,
    /// Isolated cache allocation
    Isolated,
    /// Custom policy with specific parameters
    Custom(HashMap<String, f64>),
}

/// Workload adaptation configuration
#[derive(Debug, Clone)]
pub struct WorkloadAdaptationConfig {
    /// Enable workload pattern detection
    pub pattern_detection: bool,
    /// Adaptation learning rate
    pub learning_rate: f64,
    /// Prediction model parameters
    pub prediction_model: PredictionModelParams,
    /// Workload characteristic tracking
    pub characteristic_tracking: bool,
}

/// Prediction model parameters
#[derive(Debug, Clone)]
pub struct PredictionModelParams {
    /// Model type identifier
    pub model_type: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy threshold
    pub accuracy_threshold: f64,
}

impl Default for PredictionModelParams {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("windowsize".to_string(), 10.0);
        parameters.insert("smoothing_factor".to_string(), 0.1);

        Self {
            model_type: "exponential_smoothing".to_string(),
            parameters,
            accuracy_threshold: 0.8,
        }
    }
}

/// Thread pool manager for centralized pool management
#[derive(Debug)]
pub struct ThreadPoolManager {
    /// Pool configurations by profile
    pool_configs: Arc<RwLock<HashMap<ThreadPoolProfile, AdvancedThreadPoolConfig>>>,
    /// Active thread pools
    active_pools: Arc<RwLock<HashMap<ThreadPoolProfile, Arc<AdvancedPerformanceThreadPool>>>>,
    /// Global manager instance
    #[allow(dead_code)]
    instance: Arc<Mutex<Option<Self>>>,
}

impl Default for ThreadPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadPoolManager {
    /// Create a new thread pool manager
    pub fn new() -> Self {
        Self {
            pool_configs: Arc::new(RwLock::new(HashMap::new())),
            active_pools: Arc::new(RwLock::new(HashMap::new())),
            instance: Arc::new(Mutex::new(None)),
        }
    }

    /// Get or create a thread pool for the specified profile
    pub fn get_pool(
        &self,
        profile: ThreadPoolProfile,
    ) -> LinalgResult<Arc<AdvancedPerformanceThreadPool>> {
        let pools = self.active_pools.read().expect("Operation failed");
        if let Some(pool) = pools.get(&profile) {
            return Ok(pool.clone());
        }
        drop(pools);

        // Create new pool
        let config = {
            let configs = self.pool_configs.read().expect("Operation failed");
            configs.get(&profile).cloned().unwrap_or_else(|| {
                let base_config = ThreadPoolConfig {
                    profile: profile.clone(),
                    ..Default::default()
                };
                AdvancedThreadPoolConfig {
                    base_config,
                    dynamic_sizing: DynamicSizingConfig::default(),
                    monitoring: MonitoringConfig::default(),
                    resource_isolation: ResourceIsolationConfig {
                        memory_policy: CacheAllocationPolicy::Default,
                        cpu_isolation: false,
                        memory_bandwidth_limit: None,
                        cache_partition: None,
                    },
                    workload_adaptation: WorkloadAdaptationConfig {
                        pattern_detection: true,
                        learning_rate: 0.1,
                        prediction_model: PredictionModelParams::default(),
                        characteristic_tracking: true,
                    },
                }
            })
        };

        let pool = Arc::new(AdvancedPerformanceThreadPool::new(config)?);

        let mut pools = self.active_pools.write().expect("Operation failed");
        pools.insert(profile, pool.clone());

        Ok(pool)
    }

    /// Configure a thread pool profile
    pub fn configure_profile(&self, profile: ThreadPoolProfile, config: AdvancedThreadPoolConfig) {
        let mut configs = self.pool_configs.write().expect("Operation failed");
        configs.insert(profile, config);
    }

    /// Get performance statistics for all pools
    pub fn get_all_stats(&self) -> HashMap<ThreadPoolProfile, AdvancedPerformanceStats> {
        let pools = self.active_pools.read().expect("Operation failed");
        let mut stats = HashMap::new();

        for (profile, pool) in pools.iter() {
            stats.insert(profile.clone(), pool.get_stats());
        }

        stats
    }

    /// Auto-optimize thread pool configurations based on historical performance
    pub fn auto_optimize_pools(
        &self,
    ) -> LinalgResult<Vec<(ThreadPoolProfile, AdvancedThreadPoolConfig)>> {
        let pools = self.active_pools.read().expect("Operation failed");
        let mut optimizations = Vec::new();

        for (profile, pool) in pools.iter() {
            let stats = pool.get_stats();

            // Analyze performance metrics to suggest optimizations
            let current_config = {
                let configs = self.pool_configs.read().expect("Operation failed");
                configs
                    .get(profile)
                    .cloned()
                    .unwrap_or_else(|| AdvancedThreadPoolConfig {
                        base_config: ThreadPoolConfig {
                            profile: profile.clone(),
                            ..Default::default()
                        },
                        dynamic_sizing: DynamicSizingConfig::default(),
                        monitoring: MonitoringConfig::default(),
                        resource_isolation: ResourceIsolationConfig {
                            memory_policy: CacheAllocationPolicy::Default,
                            cpu_isolation: false,
                            memory_bandwidth_limit: None,
                            cache_partition: None,
                        },
                        workload_adaptation: WorkloadAdaptationConfig {
                            pattern_detection: true,
                            learning_rate: 0.1,
                            prediction_model: PredictionModelParams::default(),
                            characteristic_tracking: true,
                        },
                    })
            };

            let mut optimized_config = current_config.clone();

            // Optimize based on performance metrics
            if stats.thread_pool_stats.total_tasks > 100 {
                // Sufficient data for optimization

                // Optimize thread count based on CPU utilization
                if stats.thread_pool_stats.thread_utilization < 0.6 {
                    // Low CPU utilization - might benefit from fewer threads to reduce overhead
                    optimized_config.base_config.active_threads =
                        (optimized_config.base_config.active_threads as f64 * 0.8) as usize;
                } else if stats.thread_pool_stats.thread_utilization > 0.9
                    && stats.thread_pool_stats.queue_length > 2
                {
                    // High CPU utilization with queuing - might benefit from more threads
                    optimized_config.base_config.active_threads =
                        (optimized_config.base_config.active_threads as f64 * 1.2) as usize;
                }

                // Optimize dynamic sizing thresholds based on performance
                if stats.thread_pool_stats.total_tasks > 500 {
                    let task_completion_variance = stats
                        .max_task_duration
                        .saturating_sub(stats.min_task_duration);
                    if task_completion_variance > Duration::from_millis(100) {
                        // High variance suggests need for more aggressive scaling
                        optimized_config.dynamic_sizing.scale_up_threshold = 0.7;
                        optimized_config.dynamic_sizing.scale_down_threshold = 0.4;
                    } else {
                        // Low variance suggests stable workload, less aggressive scaling
                        optimized_config.dynamic_sizing.scale_up_threshold = 0.85;
                        optimized_config.dynamic_sizing.scale_down_threshold = 0.25;
                    }
                }

                // Optimize affinity strategy based on workload pattern
                match profile {
                    ThreadPoolProfile::MatrixMultiplication | ThreadPoolProfile::CpuIntensive
                        // CPU-intensive tasks benefit from pinned affinity
                        if optimized_config.base_config.affinity == AffinityStrategy::None => {
                            optimized_config.base_config.affinity = AffinityStrategy::NumaSpread;
                        }
                    ThreadPoolProfile::MemoryBound => {
                        // Memory-bound tasks benefit from NUMA awareness
                        optimized_config.base_config.affinity = AffinityStrategy::NumaCompact;
                    }
                    _ => {}
                }

                // Optimize workload adaptation learning rate
                if stats.average_throughput_ops_per_sec > 1000.0 {
                    // High throughput workloads can use faster learning
                    optimized_config.workload_adaptation.learning_rate = 0.15;
                } else {
                    // Lower throughput workloads need more conservative learning
                    optimized_config.workload_adaptation.learning_rate = 0.05;
                }

                // Only suggest optimization if there's meaningful change
                if optimized_config.base_config != current_config.base_config
                    || optimized_config.dynamic_sizing.scale_up_threshold
                        != current_config.dynamic_sizing.scale_up_threshold
                    || optimized_config.workload_adaptation.learning_rate
                        != current_config.workload_adaptation.learning_rate
                {
                    optimizations.push((profile.clone(), optimized_config));
                }
            }
        }

        Ok(optimizations)
    }

    /// Apply auto-optimization suggestions
    pub fn apply_optimizations(
        &self,
        optimizations: Vec<(ThreadPoolProfile, AdvancedThreadPoolConfig)>,
    ) {
        let mut configs = self.pool_configs.write().expect("Operation failed");

        for (profile, config) in optimizations {
            configs.insert(profile, config);
        }
    }
}

/// Advanced-performance thread pool with advanced features
#[derive(Debug)]
pub struct AdvancedPerformanceThreadPool {
    /// Configuration
    #[allow(dead_code)]
    config: AdvancedThreadPoolConfig,
    /// Performance statistics
    stats: Arc<Mutex<AdvancedPerformanceStats>>,
    /// Dynamic thread manager
    thread_manager: Arc<Mutex<DynamicThreadManager>>,
    /// Workload predictor
    workload_predictor: Arc<Mutex<WorkloadPredictor>>,
    /// Performance profiler
    #[allow(dead_code)]
    profiler: Arc<Mutex<ThreadPoolProfiler>>,
}

impl AdvancedPerformanceThreadPool {
    /// Create a new advanced-performance thread pool
    pub fn new(config: AdvancedThreadPoolConfig) -> LinalgResult<Self> {
        let stats = Arc::new(Mutex::new(AdvancedPerformanceStats::default()));
        let thread_manager = Arc::new(Mutex::new(DynamicThreadManager::new(&config.base_config)?));
        let workload_predictor = Arc::new(Mutex::new(WorkloadPredictor::new()));
        let profiler = Arc::new(Mutex::new(ThreadPoolProfiler::new()));

        Ok(Self {
            config,
            stats,
            thread_manager,
            workload_predictor,
            profiler,
        })
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> AdvancedPerformanceStats {
        self.stats.lock().expect("Operation failed").clone()
    }

    /// Execute a closure with optimal thread configuration
    pub fn execute<F, R>(&self, operationtype: OperationType, task: F) -> LinalgResult<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let start_time = Instant::now();

        // Predict optimal configuration
        let predictor = self.workload_predictor.lock().expect("Operation failed");
        let predicted_characteristics = predictor.predict_workload(&operationtype);
        drop(predictor);

        // Adapt thread pool based on prediction
        {
            let mut manager = self.thread_manager.lock().expect("Operation failed");
            manager.adapt_to_workload(&predicted_characteristics)?;
        }

        // Execute the task
        let result = task();

        // Record performance metrics
        let execution_time = start_time.elapsed();
        {
            let mut stats = self.stats.lock().expect("Operation failed");
            stats.record_execution(operationtype, execution_time);
        }

        // Update workload predictor with actual performance
        {
            let mut predictor = self.workload_predictor.lock().expect("Operation failed");
            predictor.update_performance(operationtype, execution_time);
        }

        Ok(result)
    }
}

/// Dynamic thread manager for adaptive thread pool sizing
#[derive(Debug)]
pub struct DynamicThreadManager {
    /// Current configuration
    config: ThreadPoolConfig,
    /// Current thread count
    current_threads: usize,
    /// CPU utilization history
    #[allow(dead_code)]
    cpu_utilization_history: Vec<f64>,
    /// Last scaling decision time
    #[allow(dead_code)]
    last_scaling_time: Instant,
    /// Scaling decisions history
    scaling_history: Vec<ScalingDecision>,
}

impl DynamicThreadManager {
    /// Create a new dynamic thread manager
    pub fn new(config: &ThreadPoolConfig) -> LinalgResult<Self> {
        Ok(Self {
            config: config.clone(),
            current_threads: config.active_threads,
            cpu_utilization_history: Vec::new(),
            last_scaling_time: Instant::now(),
            scaling_history: Vec::new(),
        })
    }

    /// Adapt thread pool to workload characteristics
    pub fn adapt_to_workload(
        &mut self,
        characteristics: &WorkloadCharacteristics,
    ) -> LinalgResult<()> {
        let optimal_threads = self.calculate_optimal_threads(characteristics);

        if optimal_threads != self.current_threads {
            let decision = ScalingDecision {
                timestamp: Instant::now(),
                from_threads: self.current_threads,
                to_threads: optimal_threads,
                reason: ScalingReason::WorkloadAdaptation,
                confidence: characteristics.complexity_estimate,
            };

            self.scaling_history.push(decision);
            self.current_threads = optimal_threads;
        }

        Ok(())
    }

    /// Calculate optimal thread count for workload
    fn calculate_optimal_threads(&self, characteristics: &WorkloadCharacteristics) -> usize {
        let base_threads = match characteristics.pattern {
            WorkloadPattern::CpuBound => (self.config.max_threads as f64 * 0.9) as usize,
            WorkloadPattern::MemoryBound => (self.config.max_threads as f64 * 0.6) as usize,
            WorkloadPattern::Balanced => self.config.max_threads / 2,
            WorkloadPattern::IoWait => self.config.max_threads,
        };

        let complexity_factor = characteristics.complexity_estimate.min(2.0);
        let adjusted_threads = (base_threads as f64 * complexity_factor) as usize;

        adjusted_threads
            .max(self.config.min_threads)
            .min(self.config.max_threads)
    }
}

/// Workload characteristics for adaptation
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Workload pattern type
    pub pattern: WorkloadPattern,
    /// Computational complexity estimate
    pub complexity_estimate: f64,
    /// Memory usage pattern
    pub memory_usage: MemoryUsagePattern,
    /// Parallelization potential
    pub parallelization_potential: f64,
    /// Cache locality factor
    pub cache_locality: f64,
}

/// Workload pattern types
#[derive(Debug, Clone)]
pub enum WorkloadPattern {
    /// CPU-intensive computation
    CpuBound,
    /// Memory bandwidth limited
    MemoryBound,
    /// Balanced CPU and memory usage
    Balanced,
    /// I/O wait dominated
    IoWait,
}

/// Memory usage pattern
#[derive(Debug, Clone)]
pub enum MemoryUsagePattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Strided access pattern
    Strided(usize),
    /// Complex mixed pattern
    Mixed,
}

/// Operation type classification
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum OperationType {
    /// Matrix multiplication operations
    MatrixMultiplication,
    /// Matrix decomposition operations
    Decomposition(DecompositionType),
    /// Iterative solver operations
    IterativeSolver(IterativeSolverType),
    /// Vector operations
    VectorOps,
    /// Eigenvalue computations
    EigenComputation,
    /// Linear system solving
    LinearSolve,
    /// Custom operation type
    Custom(u32),
}

/// Decomposition operation types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DecompositionType {
    /// LU decomposition
    LU,
    /// QR decomposition
    QR,
    /// SVD decomposition
    SVD,
    /// Cholesky decomposition
    Cholesky,
    /// Eigenvalue decomposition
    Eigen,
    /// Schur decomposition
    Schur,
}

/// Iterative solver types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum IterativeSolverType {
    /// Conjugate Gradient
    CG,
    /// GMRES
    GMRES,
    /// BiCGSTAB
    BiCGSTAB,
    /// Jacobi method
    Jacobi,
    /// Gauss-Seidel
    GaussSeidel,
}

/// Workload predictor for performance optimization
#[derive(Debug)]
pub struct WorkloadPredictor {
    /// Historical performance data
    performance_history: HashMap<OperationType, Vec<Duration>>,
    /// Workload characteristics cache
    characteristics_cache: HashMap<OperationType, WorkloadCharacteristics>,
    /// Prediction accuracy metrics
    #[allow(dead_code)]
    prediction_accuracy: HashMap<OperationType, f64>,
}

impl Default for WorkloadPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadPredictor {
    /// Create a new workload predictor
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
            characteristics_cache: HashMap::new(),
            prediction_accuracy: HashMap::new(),
        }
    }

    /// Predict workload characteristics for an operation type
    pub fn predict_workload(&self, operationtype: &OperationType) -> WorkloadCharacteristics {
        self.characteristics_cache
            .get(operationtype)
            .cloned()
            .unwrap_or_else(|| self.default_characteristics_for_operation(operationtype))
    }

    /// Update performance data with actual execution results
    pub fn update_performance(&mut self, operation_type: OperationType, executiontime: Duration) {
        let history = self.performance_history.entry(operation_type).or_default();
        history.push(executiontime);

        // Maintain reasonable history size
        if history.len() > 100 {
            history.remove(0);
        }

        // Update characteristics based on performance patterns
        self.update_characteristics(operation_type);
    }

    /// Update workload characteristics based on performance history
    fn update_characteristics(&mut self, operationtype: OperationType) {
        let characteristics = self.default_characteristics_for_operation(&operationtype);
        self.characteristics_cache
            .insert(operationtype, characteristics);
    }

    /// Get default characteristics for an operation type
    fn default_characteristics_for_operation(
        &self,
        operation_type: &OperationType,
    ) -> WorkloadCharacteristics {
        match operation_type {
            OperationType::MatrixMultiplication => WorkloadCharacteristics {
                pattern: WorkloadPattern::CpuBound,
                complexity_estimate: 2.0,
                memory_usage: MemoryUsagePattern::Sequential,
                parallelization_potential: 0.9,
                cache_locality: 0.7,
            },
            OperationType::Decomposition(DecompositionType::LU) => WorkloadCharacteristics {
                pattern: WorkloadPattern::Balanced,
                complexity_estimate: 1.8,
                memory_usage: MemoryUsagePattern::Random,
                parallelization_potential: 0.7,
                cache_locality: 0.5,
            },
            OperationType::EigenComputation => WorkloadCharacteristics {
                pattern: WorkloadPattern::CpuBound,
                complexity_estimate: 2.5,
                memory_usage: MemoryUsagePattern::Mixed,
                parallelization_potential: 0.6,
                cache_locality: 0.4,
            },
            _ => WorkloadCharacteristics {
                pattern: WorkloadPattern::Balanced,
                complexity_estimate: 1.0,
                memory_usage: MemoryUsagePattern::Sequential,
                parallelization_potential: 0.5,
                cache_locality: 0.5,
            },
        }
    }
}

/// Thread pool performance profiler
#[derive(Debug)]
pub struct ThreadPoolProfiler {
    /// Profile metrics by operation type
    profile_metrics: HashMap<OperationType, ProfileMetrics>,
    /// Performance anomaly detection
    anomaly_detector: PerformanceAnomalyDetector,
}

impl Default for ThreadPoolProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadPoolProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            profile_metrics: HashMap::new(),
            anomaly_detector: PerformanceAnomalyDetector::new(),
        }
    }

    /// Record performance profile for an operation
    pub fn record_profile(
        &mut self,
        operation_type: OperationType,
        execution_time: Duration,
        thread_count: usize,
    ) {
        let metrics = self.profile_metrics.entry(operation_type).or_default();
        metrics.record_execution(execution_time, thread_count);

        // Check for performance anomalies
        if let Some(anomaly) = self
            .anomaly_detector
            .detect_anomaly(operation_type, execution_time)
        {
            metrics.record_anomaly(anomaly);
        }
    }

    /// Get profile metrics for an operation type
    pub fn get_metrics(&self, operationtype: &OperationType) -> Option<&ProfileMetrics> {
        self.profile_metrics.get(operationtype)
    }
}

/// Profile metrics for performance analysis
#[derive(Debug)]
pub struct ProfileMetrics {
    /// Total executions recorded
    pub total_executions: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Execution time variance
    pub execution_time_variance: f64,
    /// Optimal thread count
    pub optimal_thread_count: usize,
    /// Performance anomalies
    pub anomalies: Vec<PerformanceAnomaly>,
}

impl Default for ProfileMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileMetrics {
    /// Create new profile metrics
    pub fn new() -> Self {
        Self {
            total_executions: 0,
            avg_execution_time: Duration::ZERO,
            execution_time_variance: 0.0,
            optimal_thread_count: 1,
            anomalies: Vec::new(),
        }
    }

    /// Record an execution
    pub fn record_execution(&mut self, execution_time: Duration, threadcount: usize) {
        let old_avg = self.avg_execution_time;
        self.total_executions += 1;

        // Update average using incremental formula
        let n = self.total_executions as f64;
        let new_time_ms = execution_time.as_secs_f64() * 1000.0;
        let old_avg_ms = old_avg.as_secs_f64() * 1000.0;
        let new_avg_ms = old_avg_ms + (new_time_ms - old_avg_ms) / n;

        self.avg_execution_time = Duration::from_secs_f64(new_avg_ms / 1000.0);

        // Update variance
        let diff = new_time_ms - new_avg_ms;
        self.execution_time_variance = ((n - 1.0) * self.execution_time_variance + diff * diff) / n;

        // Update optimal thread _count heuristic
        if execution_time < self.avg_execution_time {
            self.optimal_thread_count = threadcount;
        }
    }

    /// Record a performance anomaly
    pub fn record_anomaly(&mut self, anomaly: PerformanceAnomaly) {
        self.anomalies.push(anomaly);

        // Maintain reasonable anomaly history
        if self.anomalies.len() > 50 {
            self.anomalies.remove(0);
        }
    }
}

/// Performance anomaly detector
#[derive(Debug)]
pub struct PerformanceAnomalyDetector {
    /// Baseline performance expectations
    baselines: HashMap<OperationType, Duration>,
    /// Anomaly sensitivity threshold
    sensitivity_threshold: f64,
}

impl Default for PerformanceAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAnomalyDetector {
    /// Create a new anomaly detector
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            sensitivity_threshold: 2.0, // 2x slower than baseline
        }
    }

    /// Detect performance anomaly
    pub fn detect_anomaly(
        &mut self,
        operation_type: OperationType,
        execution_time: Duration,
    ) -> Option<PerformanceAnomaly> {
        if let Some(&baseline) = self.baselines.get(&operation_type) {
            let ratio = execution_time.as_secs_f64() / baseline.as_secs_f64();

            if ratio > self.sensitivity_threshold {
                return Some(PerformanceAnomaly {
                    operation_type,
                    severity: if ratio > 5.0 {
                        AnomalySeverity::Critical
                    } else {
                        AnomalySeverity::Warning
                    },
                    anomaly_type: AnomalyType::SlowExecution,
                    measured_time: execution_time,
                    expected_time: baseline,
                    deviation_factor: ratio,
                });
            }
        } else {
            // Set baseline for first measurement
            self.baselines.insert(operation_type, execution_time);
        }

        None
    }
}

/// Performance anomaly information
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Operation type that experienced anomaly
    pub operation_type: OperationType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Measured execution time
    pub measured_time: Duration,
    /// Expected execution time
    pub expected_time: Duration,
    /// Deviation factor from expected
    pub deviation_factor: f64,
}

/// Anomaly severity levels
#[derive(Debug, Clone)]
pub enum AnomalySeverity {
    /// Information level
    Info,
    /// Warning level
    Warning,
    /// Critical level
    Critical,
}

/// Anomaly type classification
#[derive(Debug, Clone)]
pub enum AnomalyType {
    /// Execution time significantly slower than expected
    SlowExecution,
    /// High memory usage
    HighMemoryUsage,
    /// CPU utilization anomaly
    CpuUtilizationAnomaly,
    /// Thread contention detected
    ThreadContention,
}

/// Scaling decision record
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    /// When the decision was made
    pub timestamp: Instant,
    /// Previous thread count
    pub from_threads: usize,
    /// New thread count
    pub to_threads: usize,
    /// Reason for scaling
    pub reason: ScalingReason,
    /// Confidence in the decision
    pub confidence: f64,
}

/// Reasons for scaling decisions
#[derive(Debug, Clone)]
pub enum ScalingReason {
    /// High CPU utilization
    HighCpuUtilization,
    /// Low CPU utilization
    LowCpuUtilization,
    /// Workload adaptation
    WorkloadAdaptation,
    /// Performance optimization
    PerformanceOptimization,
    /// Resource constraints
    ResourceConstraints,
}

/// Comprehensive performance statistics
#[derive(Debug, Clone)]
pub struct AdvancedPerformanceStats {
    /// Basic thread pool statistics
    pub thread_pool_stats: ThreadPoolStats,
    /// Memory usage metrics
    pub memory_metrics: MemoryMetrics,
    /// Resource usage patterns
    pub resource_patterns: HashMap<OperationType, ResourceUsagePattern>,
    /// Performance trends
    pub performance_trends: HashMap<OperationType, Vec<f64>>,
    /// Maximum task duration
    pub max_task_duration: Duration,
    /// Minimum task duration
    pub min_task_duration: Duration,
    /// Average throughput in operations per second
    pub average_throughput_ops_per_sec: f64,
}

impl Default for AdvancedPerformanceStats {
    fn default() -> Self {
        Self {
            thread_pool_stats: ThreadPoolStats::default(),
            memory_metrics: MemoryMetrics::default(),
            resource_patterns: HashMap::new(),
            performance_trends: HashMap::new(),
            max_task_duration: Duration::from_millis(0),
            min_task_duration: Duration::from_millis(u64::MAX),
            average_throughput_ops_per_sec: 0.0,
        }
    }
}

impl AdvancedPerformanceStats {
    /// Record execution performance
    pub fn record_execution(&mut self, operation_type: OperationType, executiontime: Duration) {
        self.thread_pool_stats.total_tasks += 1;
        self.thread_pool_stats.total_execution_time += executiontime;

        // Update performance trends
        let trends = self.performance_trends.entry(operation_type).or_default();
        trends.push(executiontime.as_secs_f64() * 1000.0); // Store as milliseconds

        // Maintain reasonable trend history
        if trends.len() > 100 {
            trends.remove(0);
        }
    }
}

/// Basic thread pool statistics
#[derive(Debug, Clone, Default)]
pub struct ThreadPoolStats {
    /// Total tasks executed
    pub total_tasks: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Active threads count
    pub active_threads: usize,
    /// Queue length
    pub queue_length: usize,
    /// Thread utilization percentage
    pub thread_utilization: f64,
}

/// Memory usage metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    pub current_usage: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Average memory usage in bytes
    pub average_usage: u64,
    /// Memory allocation rate (bytes per second)
    pub allocation_rate: f64,
}

/// Resource usage pattern for optimization
#[derive(Debug, Clone)]
pub struct ResourceUsagePattern {
    /// CPU usage pattern
    pub cpu_pattern: Vec<f64>,
    /// Memory usage pattern
    pub memory_pattern: Vec<u64>,
    /// I/O pattern (if applicable)
    pub io_pattern: Option<Vec<f64>>,
    /// Cache miss rate pattern
    pub cache_miss_pattern: Option<Vec<f64>>,
}

/// Scoped thread pool for temporary operations
#[derive(Debug)]
pub struct ScopedThreadPool {
    /// Base configuration
    #[allow(dead_code)]
    config: ThreadPoolConfig,
    /// Creation timestamp
    created_at: Instant,
    /// Automatic cleanup timeout
    cleanup_timeout: Duration,
}

impl ScopedThreadPool {
    /// Create a new scoped thread pool
    pub fn new(_config: ThreadPoolConfig, cleanuptimeout: Duration) -> Self {
        Self {
            config: _config,
            created_at: Instant::now(),
            cleanup_timeout: cleanuptimeout,
        }
    }

    /// Check if the pool should be cleaned up
    pub fn should_cleanup(&self) -> bool {
        self.created_at.elapsed() > self.cleanup_timeout
    }
}

/// Global thread pool manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<Arc<Mutex<ThreadPoolManager>>> =
    std::sync::OnceLock::new();

/// Get the global thread pool manager
#[allow(dead_code)]
pub fn get_global_manager() -> Arc<Mutex<ThreadPoolManager>> {
    GLOBAL_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(ThreadPoolManager::new())))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_config_default() {
        let config = ThreadPoolConfig::default();
        assert!(config.max_threads > 0);
        assert!(config.min_threads > 0);
        assert!(config.min_threads <= config.max_threads);
    }

    #[test]
    fn test_dynamic_sizing_config() {
        let config = DynamicSizingConfig::default();
        assert!(config.scale_up_threshold > config.scale_down_threshold);
        assert!(config.max_scaling_factor > 1.0);
    }

    #[test]
    fn test_workload_predictor() {
        let mut predictor = WorkloadPredictor::new();
        let characteristics = predictor.predict_workload(&OperationType::MatrixMultiplication);

        assert!(matches!(characteristics.pattern, WorkloadPattern::CpuBound));
        assert!(characteristics.complexity_estimate > 0.0);

        predictor.update_performance(
            OperationType::MatrixMultiplication,
            Duration::from_millis(100),
        );

        let updated_characteristics =
            predictor.predict_workload(&OperationType::MatrixMultiplication);
        assert!(matches!(
            updated_characteristics.pattern,
            WorkloadPattern::CpuBound
        ));
    }

    #[test]
    fn test_performance_anomaly_detection() {
        let mut detector = PerformanceAnomalyDetector::new();

        // First measurement sets baseline
        let baseline = Duration::from_millis(100);
        assert!(detector
            .detect_anomaly(OperationType::MatrixMultiplication, baseline)
            .is_none());

        // Slow execution should trigger anomaly
        let slow_execution = Duration::from_millis(300);
        let anomaly = detector.detect_anomaly(OperationType::MatrixMultiplication, slow_execution);
        assert!(anomaly.is_some());

        if let Some(anomaly) = anomaly {
            assert_eq!(anomaly.operation_type, OperationType::MatrixMultiplication);
            assert!(matches!(anomaly.anomaly_type, AnomalyType::SlowExecution));
        }
    }

    #[test]
    fn test_thread_pool_manager() {
        let manager = ThreadPoolManager::new();

        // Test pool creation
        let pool = manager.get_pool(ThreadPoolProfile::LinearAlgebra);
        assert!(pool.is_ok());

        // Test configuration
        let config = AdvancedThreadPoolConfig {
            base_config: ThreadPoolConfig::default(),
            dynamic_sizing: DynamicSizingConfig::default(),
            monitoring: MonitoringConfig::default(),
            resource_isolation: ResourceIsolationConfig {
                memory_policy: CacheAllocationPolicy::Default,
                cpu_isolation: false,
                memory_bandwidth_limit: None,
                cache_partition: None,
            },
            workload_adaptation: WorkloadAdaptationConfig {
                pattern_detection: true,
                learning_rate: 0.1,
                prediction_model: PredictionModelParams::default(),
                characteristic_tracking: true,
            },
        };

        manager.configure_profile(ThreadPoolProfile::MatrixMultiplication, config);

        let stats = manager.get_all_stats();
        assert!(stats.contains_key(&ThreadPoolProfile::LinearAlgebra));
    }
}
