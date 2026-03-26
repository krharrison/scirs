/// NUMA-aware work stealing: type definitions.
///
/// This module contains all fundamental data structures for the NUMA-aware
/// work-stealing scheduler, including topology descriptions, task representations,
/// configuration, and statistics.

/// Information about a single CPU core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreInfo {
    /// Logical core identifier.
    pub core_id: usize,
    /// NUMA node this core belongs to.
    pub numa_node: usize,
    /// L3 cache size in kilobytes shared by cores on this NUMA node.
    pub cache_size_kb: usize,
}

/// Simulated NUMA topology — n_nodes × n_cores_per_node.
///
/// In a pure-Rust environment we cannot query the OS directly, so topology
/// is either supplied explicitly via [`NumaTopology::from_config`] or
/// defaults to 2 NUMA nodes × 4 cores.
#[derive(Debug, Clone, Default)]
pub struct NumaTopology {
    /// Number of NUMA nodes.
    pub n_nodes: usize,
    /// Number of physical cores per NUMA node.
    pub n_cores_per_node: usize,
    /// Flat list of all cores in node-major order.
    pub cores: Vec<CoreInfo>,
}

impl NumaTopology {
    /// Construct a topology with the specified node/core counts.
    ///
    /// Each NUMA node is assigned a simulated 8 MiB shared L3 cache
    /// (8192 KiB).  Core IDs are numbered 0..(n_nodes * n_cores_per_node)
    /// in row-major (node-major) order.
    pub fn from_config(nodes: usize, cores_per_node: usize) -> Self {
        let mut cores = Vec::with_capacity(nodes * cores_per_node);
        for node in 0..nodes {
            for local in 0..cores_per_node {
                cores.push(CoreInfo {
                    core_id: node * cores_per_node + local,
                    numa_node: node,
                    cache_size_kb: 8192,
                });
            }
        }
        Self {
            n_nodes: nodes,
            n_cores_per_node: cores_per_node,
            cores,
        }
    }
}

/// Configuration for the NUMA-aware work-stealing scheduler.
///
/// Derives `Default` so callers only need to override fields they care about.
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Total number of worker threads to spawn.
    pub n_workers: usize,
    /// Minimum work-items in a queue before it becomes a steal candidate.
    pub steal_threshold: usize,
    /// Initial capacity of each worker's local deque.
    pub local_queue_size: usize,
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            steal_threshold: 1,
            local_queue_size: 64,
        }
    }
}

/// A unit of work submitted to the scheduler.
///
/// The generic parameter `T` is the return type of the closure.
/// `affinity` optionally pins the task to a preferred NUMA node.
pub struct Task<T> {
    /// The closure to execute.  Boxed to allow type-erasure.
    pub work: Box<dyn FnOnce() -> T + Send>,
    /// Scheduling priority — higher values run first (not yet implemented in
    /// the deque ordering, reserved for future use).
    pub priority: u8,
    /// If `Some(node)`, the scheduler will prefer to place this task on a
    /// worker that lives in the given NUMA node.
    pub affinity: Option<usize>,
}

impl<T> std::fmt::Debug for Task<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("priority", &self.priority)
            .field("affinity", &self.affinity)
            .finish()
    }
}

impl<T: Send + 'static> Task<T> {
    /// Create a new task with default priority and no NUMA affinity.
    pub fn new(work: impl FnOnce() -> T + Send + 'static) -> Self {
        Self {
            work: Box::new(work),
            priority: 0,
            affinity: None,
        }
    }

    /// Attach a preferred NUMA node to this task.
    pub fn with_affinity(mut self, numa_node: usize) -> Self {
        self.affinity = Some(numa_node);
        self
    }

    /// Set the scheduling priority (higher = more urgent).
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// The outcome of executing a single task.
#[derive(Debug, Clone)]
pub struct TaskResult<T> {
    /// Value produced by the task closure.
    pub result: T,
    /// ID of the worker thread that executed the task.
    pub worker_id: usize,
    /// `true` if the task was stolen from another worker's queue.
    pub stolen: bool,
}

/// Aggregate statistics for a running scheduler instance.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of tasks that have completed execution.
    pub tasks_executed: u64,
    /// Number of tasks executed after being stolen from another worker.
    pub tasks_stolen: u64,
    /// Total steal attempts (successful or not).
    pub steal_attempts: u64,
    /// Ratio of tasks executed on their preferred NUMA node (0.0–1.0).
    pub locality_score: f64,
}
