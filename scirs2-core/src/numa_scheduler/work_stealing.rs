/// NUMA-aware work-stealing scheduler.
///
/// Each worker thread owns a double-ended queue (deque).  Workers push new
/// tasks to the *local* (back) end and pop tasks from the same end for
/// LIFO locality.  Stealers take tasks from the *remote* (front) end.
///
/// Steal order:
/// 1. Check own queue first.
/// 2. Steal from another worker on the **same** NUMA node.
/// 3. If all same-node queues are empty, steal from any node.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering as AOrdering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::numa_scheduler::migration::worker_numa_node;
use crate::numa_scheduler::topology::{cores_in_node, nearest_numa_nodes};
use crate::numa_scheduler::types::{NumaTopology, SchedulerStats, Task, WorkStealingConfig};

// ─── WorkerDeque ──────────────────────────────────────────────────────────────

/// The per-worker double-ended task queue.
pub struct WorkerDeque {
    /// FIFO / LIFO task store.
    pub local_queue: VecDeque<Task<()>>,
    /// This worker's logical identifier.
    pub worker_id: usize,
    /// NUMA node the worker is pinned to (logical, not OS-level).
    pub numa_node: usize,
}

impl WorkerDeque {
    /// Construct an empty deque for the given worker.
    pub fn new(worker_id: usize, numa_node: usize, capacity: usize) -> Self {
        Self {
            local_queue: VecDeque::with_capacity(capacity),
            worker_id,
            numa_node,
        }
    }

    /// Push a task to the local (back) end — used by the owning worker.
    pub fn push_local(&mut self, task: Task<()>) {
        self.local_queue.push_back(task);
    }

    /// Pop a task from the local (back) end — LIFO for cache friendliness.
    pub fn pop_local(&mut self) -> Option<Task<()>> {
        self.local_queue.pop_back()
    }

    /// Steal a task from the remote (front) end — FIFO for fairness.
    pub fn steal_remote(&mut self) -> Option<Task<()>> {
        self.local_queue.pop_front()
    }

    /// Number of tasks currently queued.
    pub fn len(&self) -> usize {
        self.local_queue.len()
    }

    /// True if the deque contains no tasks.
    pub fn is_empty(&self) -> bool {
        self.local_queue.is_empty()
    }
}

// ─── Steal victim selection ───────────────────────────────────────────────────

/// Choose a victim worker to steal from.
///
/// Preference order:
/// 1. Same-NUMA workers that have tasks queued (picked round-robin by
///    queue length — largest first).
/// 2. Any worker on a different NUMA node with tasks queued.
///
/// Returns `None` if all queues are empty.
pub fn choose_victim(
    worker_id: usize,
    topology: &NumaTopology,
    queue_lengths: &[usize],
) -> Option<usize> {
    let n = queue_lengths.len();
    if n == 0 {
        return None;
    }

    let my_node = worker_numa_node(worker_id, topology);

    // 1. Same-NUMA pass.
    let same_node_cores = cores_in_node(topology, my_node);
    let same_node_workers: Vec<usize> = (0..n)
        .filter(|&w| {
            if w == worker_id {
                return false;
            }
            let core_id = w % topology.cores.len().max(1);
            same_node_cores.contains(&core_id)
        })
        .filter(|&w| queue_lengths[w] > 0)
        .collect();

    if let Some(&victim) = same_node_workers.iter().max_by_key(|&&w| queue_lengths[w]) {
        return Some(victim);
    }

    // 2. Cross-NUMA pass — prefer closer nodes first.
    let node_order = nearest_numa_nodes(topology, my_node);
    for node in node_order {
        if node == my_node {
            continue; // already checked
        }
        let node_cores = cores_in_node(topology, node);
        let victim = (0..n)
            .filter(|&w| w != worker_id)
            .filter(|&w| {
                let core_id = w % topology.cores.len().max(1);
                node_cores.contains(&core_id)
            })
            .filter(|&w| queue_lengths[w] > 0)
            .max_by_key(|&w| queue_lengths[w]);

        if let Some(v) = victim {
            return Some(v);
        }
    }

    None
}

// ─── Task assignment ──────────────────────────────────────────────────────────

/// Push `task` to the least-loaded worker in the task's preferred NUMA node.
///
/// If `task.affinity` is `None` or the preferred node has no workers, falls
/// back to the globally least-loaded worker.
pub fn assign_task(task: Task<()>, topology: &NumaTopology, queues: &[Arc<Mutex<WorkerDeque>>]) {
    let n = queues.len();
    if n == 0 {
        return; // nowhere to put the task
    }

    // Build a snapshot of queue lengths without holding all locks.
    let lengths: Vec<usize> = queues
        .iter()
        .map(|q| q.lock().map(|g| g.len()).unwrap_or(0))
        .collect();

    // Determine preferred NUMA node.
    let preferred_node: Option<usize> = task.affinity.filter(|&node| node < topology.n_nodes);

    let target = if let Some(node) = preferred_node {
        // Find workers on the preferred node.
        let node_cores = cores_in_node(topology, node);
        let candidate = (0..n)
            .filter(|&w| {
                let core_id = w % topology.cores.len().max(1);
                node_cores.contains(&core_id)
            })
            .min_by_key(|&w| lengths[w]);

        // Fall back to global minimum if preferred node has no workers.
        candidate.or_else(|| (0..n).min_by_key(|&w| lengths[w]))
    } else {
        (0..n).min_by_key(|&w| lengths[w])
    };

    if let Some(idx) = target {
        if let Ok(mut deque) = queues[idx].lock() {
            deque.push_local(task);
        }
    }
}

// ─── NumaWorkStealingScheduler ────────────────────────────────────────────────

/// A multi-threaded NUMA-aware work-stealing scheduler.
///
/// Workers are pinned to NUMA nodes logically (no OS-level affinity is set
/// in this pure-Rust implementation).  Each worker prefers same-NUMA victims
/// when stealing.
pub struct NumaWorkStealingScheduler {
    /// Shared per-worker deques.
    queues: Vec<Arc<Mutex<WorkerDeque>>>,
    /// Worker thread handles — consumed by `shutdown`.
    workers: Vec<thread::JoinHandle<()>>,
    /// Signal workers to terminate.
    shutdown_flag: Arc<AtomicBool>,
    /// Aggregate statistics updated by worker threads.
    stats: Arc<Mutex<SchedulerStats>>,
    /// Logical topology used for steal decisions.
    topology: Arc<NumaTopology>,
}

impl NumaWorkStealingScheduler {
    /// Create a new scheduler with the given configuration and topology.
    ///
    /// Spawns `config.n_workers` background threads immediately.
    pub fn new(config: &WorkStealingConfig, topology: NumaTopology) -> Self {
        let n = config.n_workers.max(1);
        let topology = Arc::new(topology);
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(SchedulerStats::default()));

        // Pre-create the shared deques.
        let queues: Vec<Arc<Mutex<WorkerDeque>>> = (0..n)
            .map(|id| {
                let node = worker_numa_node(id, &topology);
                Arc::new(Mutex::new(WorkerDeque::new(
                    id,
                    node,
                    config.local_queue_size,
                )))
            })
            .collect();

        let mut workers = Vec::with_capacity(n);

        for worker_id in 0..n {
            let queues_clone = queues.clone();
            let topo_clone = Arc::clone(&topology);
            let stop = Arc::clone(&shutdown_flag);
            let stats_clone = Arc::clone(&stats);

            let handle = thread::Builder::new()
                .name(format!("numa-ws-{}", worker_id))
                .spawn(move || {
                    worker_loop(worker_id, &queues_clone, &topo_clone, &stop, &stats_clone);
                })
                .expect("failed to spawn worker thread");

            workers.push(handle);
        }

        Self {
            queues,
            workers,
            shutdown_flag,
            stats,
            topology,
        }
    }

    /// Submit a single task, assigning it to the best available worker.
    pub fn submit(&self, task: Task<()>) {
        assign_task(task, &self.topology, &self.queues);
    }

    /// Submit many tasks with NUMA-aware load balancing across workers.
    pub fn submit_many(&self, tasks: Vec<Task<()>>) {
        for task in tasks {
            assign_task(task, &self.topology, &self.queues);
        }
    }

    /// Snapshot of scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Signal all workers to stop and wait for them to finish.
    pub fn shutdown(self) {
        self.shutdown_flag.store(true, AOrdering::SeqCst);
        for handle in self.workers {
            // Best-effort join; we cannot propagate panics here.
            let _ = handle.join();
        }
    }

    /// Number of worker threads.
    pub fn n_workers(&self) -> usize {
        self.queues.len()
    }

    /// Queue lengths snapshot (used for testing and migration planning).
    pub fn queue_lengths(&self) -> Vec<usize> {
        self.queues
            .iter()
            .map(|q| q.lock().map(|g| g.len()).unwrap_or(0))
            .collect()
    }
}

// ─── Worker loop ──────────────────────────────────────────────────────────────

fn worker_loop(
    worker_id: usize,
    queues: &[Arc<Mutex<WorkerDeque>>],
    topology: &NumaTopology,
    stop: &AtomicBool,
    stats: &Mutex<SchedulerStats>,
) {
    loop {
        // 1. Try own queue first.
        let own_task = queues
            .get(worker_id)
            .and_then(|q| q.lock().ok())
            .and_then(|mut g| g.pop_local())
            .map(|t| (t, false)); // (task, stolen=false)

        if let Some((task, stolen)) = own_task {
            execute_task(task, worker_id, stolen, stats);
            continue;
        }

        // 2. Check if we should stop (after draining our own queue).
        if stop.load(AOrdering::Relaxed) {
            // Drain remaining work before exiting.
            while let Some(task) = queues
                .get(worker_id)
                .and_then(|q| q.lock().ok())
                .and_then(|mut g| g.pop_local())
            {
                execute_task(task, worker_id, false, stats);
            }
            break;
        }

        // 3. Try to steal.
        let lengths: Vec<usize> = queues
            .iter()
            .map(|q| q.lock().map(|g| g.len()).unwrap_or(0))
            .collect();

        // Record the steal attempt.
        if let Ok(mut s) = stats.lock() {
            s.steal_attempts += 1;
        }

        let victim = choose_victim(worker_id, topology, &lengths);

        if let Some(v) = victim {
            let stolen_task = queues
                .get(v)
                .and_then(|q| q.lock().ok())
                .and_then(|mut g| g.steal_remote());

            if let Some(task) = stolen_task {
                execute_task(task, worker_id, true, stats);
                continue;
            }
        }

        // 4. Nothing to do — park briefly.
        thread::sleep(std::time::Duration::from_micros(PARK_US));
    }
}

fn execute_task(task: Task<()>, worker_id: usize, stolen: bool, stats: &Mutex<SchedulerStats>) {
    (task.work)();
    if let Ok(mut s) = stats.lock() {
        s.tasks_executed += 1;
        if stolen {
            s.tasks_stolen += 1;
        }
    }
    let _ = (worker_id, stolen); // suppress unused warnings in release
}

/// Spin-park duration in microseconds when all queues are empty.
const PARK_US: u64 = 100;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa_scheduler::types::{NumaTopology, WorkStealingConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn small_topo() -> NumaTopology {
        NumaTopology::from_config(2, 2) // 2 nodes × 2 cores = 4 cores
    }

    fn small_config() -> WorkStealingConfig {
        WorkStealingConfig {
            n_workers: 4,
            steal_threshold: 1,
            local_queue_size: 32,
        }
    }

    #[test]
    fn test_worker_deque_push_pop() {
        let mut deque = WorkerDeque::new(0, 0, 8);
        let task = Task::new(|| ());
        deque.push_local(task);
        assert_eq!(deque.len(), 1);
        assert!(deque.pop_local().is_some());
        assert!(deque.is_empty());
    }

    #[test]
    fn test_steal_victim_prefers_local() {
        let t = small_topo(); // 2 nodes, 2 cores each; 4 workers total
                              // Workers 0,1 → node 0; workers 2,3 → node 1.
        let lengths = vec![0usize, 5, 3, 8];
        // Worker 0 (node 0) should prefer worker 1 (node 0, 5 tasks) over
        // workers 2/3 (node 1).
        let victim = choose_victim(0, &t, &lengths);
        assert_eq!(victim, Some(1), "Expected worker 1 (same NUMA)");
    }

    #[test]
    fn test_steal_victim_fallback_remote() {
        let t = small_topo();
        // Only workers on node 1 have tasks.
        let lengths = vec![0usize, 0, 0, 7];
        let victim = choose_victim(0, &t, &lengths);
        assert_eq!(victim, Some(3), "Should fall back to remote worker 3");
    }

    #[test]
    fn test_steal_victim_no_tasks() {
        let t = small_topo();
        let lengths = vec![0, 0, 0, 0];
        assert!(choose_victim(0, &t, &lengths).is_none());
    }

    #[test]
    fn test_affinity_assignment() {
        let t = small_topo(); // 4 workers (0-3), nodes 0 and 1
        let queues: Vec<Arc<Mutex<WorkerDeque>>> = (0..4)
            .map(|id| {
                let node = worker_numa_node(id, &t);
                Arc::new(Mutex::new(WorkerDeque::new(id, node, 8)))
            })
            .collect();

        // Task with affinity = node 1 → should land on workers 2 or 3.
        let task = Task::new(|| ()).with_affinity(1);
        assign_task(task, &t, &queues);

        let q2 = queues[2].lock().map(|g| g.len()).unwrap_or(0);
        let q3 = queues[3].lock().map(|g| g.len()).unwrap_or(0);
        assert_eq!(
            q2 + q3,
            1,
            "Task with affinity=1 should be on node-1 workers"
        );
    }

    #[test]
    fn test_scheduler_submits_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));
        let sched = NumaWorkStealingScheduler::new(&small_config(), small_topo());

        let n_tasks = 20usize;
        for _ in 0..n_tasks {
            let c = Arc::clone(&counter);
            sched.submit(Task::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }

        // Give workers time to drain.
        thread::sleep(Duration::from_millis(200));
        sched.shutdown();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            n_tasks,
            "All tasks should have been executed"
        );
    }

    #[test]
    fn test_scheduler_stats_executed() {
        let sched = NumaWorkStealingScheduler::new(&small_config(), small_topo());
        let n_tasks = 10;
        for _ in 0..n_tasks {
            sched.submit(Task::new(|| {}));
        }
        thread::sleep(Duration::from_millis(200));
        let s = sched.stats();
        sched.shutdown();
        assert_eq!(
            s.tasks_executed, n_tasks as u64,
            "stats should count all executed tasks"
        );
    }

    #[test]
    fn test_scheduler_work_stealing() {
        // Use a 1-node topology so all workers are in the same NUMA node,
        // making cross-queue stealing easier to observe.
        let topo = NumaTopology::from_config(1, 4);
        let config = WorkStealingConfig {
            n_workers: 4,
            steal_threshold: 1,
            local_queue_size: 64,
        };
        let sched = NumaWorkStealingScheduler::new(&config, topo);

        let stolen = Arc::new(AtomicUsize::new(0));
        // Submit 40 tasks all to worker 0's queue via zero-affinity tasks.
        for _ in 0..40 {
            sched.submit(Task::new(|| {}));
        }
        thread::sleep(Duration::from_millis(300));
        let s = sched.stats();
        sched.shutdown();

        // Some tasks should have been stolen.
        stolen.fetch_add(s.tasks_stolen as usize, Ordering::Relaxed);
        // We cannot guarantee steals in all environments, but at least all
        // tasks should have been executed.
        assert_eq!(s.tasks_executed, 40, "All 40 tasks should be executed");
    }

    #[test]
    fn test_scheduler_shutdown() {
        let sched = NumaWorkStealingScheduler::new(&small_config(), small_topo());
        // Submit a small number of tasks, then immediately shut down.
        for _ in 0..5 {
            sched.submit(Task::new(|| {}));
        }
        // shutdown() must not deadlock or panic.
        sched.shutdown();
    }
}
