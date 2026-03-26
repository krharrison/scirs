/// NUMA-aware work-stealing scheduler with topology-aware task migration.
///
/// # Overview
///
/// This module provides a pure-Rust implementation of a NUMA-aware
/// work-stealing thread pool.  Key components:
///
/// * [`types`] — data structures (`NumaTopology`, `Task`, `SchedulerStats`, …)
/// * [`topology`] — NUMA topology detection/queries (simulated, pure Rust)
/// * [`work_stealing`] — scheduler, per-worker deques, steal-victim selection
/// * [`migration`] — load-imbalance detection and migration planning
///
/// # Example
///
/// ```rust
/// use scirs2_core::numa_scheduler::{
///     types::{NumaTopology, WorkStealingConfig, Task},
///     topology::detect_topology,
///     work_stealing::NumaWorkStealingScheduler,
/// };
/// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
///
/// let topo = detect_topology();
/// let config = WorkStealingConfig::default();
/// let sched = NumaWorkStealingScheduler::new(&config, topo);
///
/// let counter = Arc::new(AtomicUsize::new(0));
/// for _ in 0..10 {
///     let c = Arc::clone(&counter);
///     sched.submit(Task::new(move || { c.fetch_add(1, Ordering::Relaxed); }));
/// }
/// std::thread::sleep(std::time::Duration::from_millis(100));
/// sched.shutdown();
/// ```
pub mod migration;
pub mod topology;
pub mod types;
pub mod work_stealing;

// Convenience re-exports.
pub use migration::{load_imbalance, migration_cost, migration_plan, should_migrate};
pub use topology::{
    cache_distance, cores_in_node, detect_topology, distance, nearest_numa_nodes, node_of_core,
    shared_l3_cores,
};
pub use types::{CoreInfo, NumaTopology, SchedulerStats, Task, TaskResult, WorkStealingConfig};
pub use work_stealing::{assign_task, choose_victim, NumaWorkStealingScheduler, WorkerDeque};
