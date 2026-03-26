/// Topology-aware task migration logic.
///
/// This module decides *when* and *where* to move tasks between worker
/// queues so that load is balanced while respecting NUMA locality.
/// Migration is intentionally conservative: we never migrate more tasks
/// than necessary, and cross-NUMA migration incurs a higher penalty than
/// intra-NUMA migration.
use crate::numa_scheduler::topology::{cores_in_node, node_of_core};
use crate::numa_scheduler::types::NumaTopology;

// ─── Imbalance measurement ────────────────────────────────────────────────────

/// Coefficient of variation of `queue_lengths`.
///
/// CoV = σ / μ, where σ is the population standard deviation and μ is the
/// mean.  Returns 0.0 if the mean is zero (all queues empty — perfectly
/// balanced).
pub fn load_imbalance(queue_lengths: &[usize]) -> f64 {
    if queue_lengths.is_empty() {
        return 0.0;
    }
    let n = queue_lengths.len() as f64;
    let mean = queue_lengths.iter().sum::<usize>() as f64 / n;
    if mean < f64::EPSILON {
        return 0.0;
    }
    let variance = queue_lengths
        .iter()
        .map(|&q| {
            let diff = q as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    variance.sqrt() / mean
}

/// Return `true` when the measured imbalance exceeds `threshold`.
pub fn should_migrate(imbalance: f64, threshold: f64) -> bool {
    imbalance > threshold
}

// ─── Migration cost model ─────────────────────────────────────────────────────

/// Estimated cost of moving `task_count` tasks from worker `from` to worker
/// `to`.
///
/// * Same NUMA node: cost factor 1.0 (cheap, no cross-NUMA bus traffic).
/// * Different NUMA nodes: cost factor 2.5 (expensive).
///
/// The final cost is `factor × task_count`.
pub fn migration_cost(
    from_worker: usize,
    to_worker: usize,
    topology: &NumaTopology,
    task_count: usize,
) -> f64 {
    let node_from = worker_numa_node(from_worker, topology);
    let node_to = worker_numa_node(to_worker, topology);
    let factor = if node_from == node_to {
        1.0_f64
    } else {
        2.5_f64
    };
    factor * task_count as f64
}

// ─── Migration planning ───────────────────────────────────────────────────────

/// Plan a minimal set of migrations to rebalance queue lengths.
///
/// Returns a list of `(from_worker, to_worker, n_tasks)` tuples, meaning
/// "move `n_tasks` items from `from_worker`'s queue to `to_worker`'s queue".
///
/// Strategy:
/// 1. Detect overloaded workers (above mean) and underloaded workers (below
///    mean).
/// 2. For each overloaded worker, prefer moving tasks to an underloaded
///    worker on the **same NUMA node** first.
/// 3. Only cross NUMA boundaries when no intra-NUMA candidate exists.
/// 4. Never migrate if the migration cost exceeds the expected speedup
///    (`expected_speedup` = tasks_moved × 1 unit; cost as defined above).
pub fn migration_plan(
    queue_lengths: &[usize],
    topology: &NumaTopology,
) -> Vec<(usize, usize, usize)> {
    if queue_lengths.is_empty() {
        return Vec::new();
    }

    let n = queue_lengths.len();
    let total: usize = queue_lengths.iter().sum();
    let mean = total as f64 / n as f64;

    // Mutable copy we can adjust as we record planned moves.
    let mut lengths: Vec<f64> = queue_lengths.iter().map(|&q| q as f64).collect();

    let mut plan: Vec<(usize, usize, usize)> = Vec::new();

    // Collect initially overloaded / underloaded workers.
    // We iterate until no more beneficial moves exist.
    let max_rounds = n * 2;
    for _ in 0..max_rounds {
        // Find the most overloaded worker.
        let maybe_from = (0..n)
            .filter(|&w| lengths[w] > mean + 0.5)
            .max_by(|&a, &b| {
                lengths[a]
                    .partial_cmp(&lengths[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let from = match maybe_from {
            Some(w) => w,
            None => break,
        };

        // Find a suitable underloaded target.
        let node_from = worker_numa_node(from, topology);

        // Prefer same-NUMA underloaded workers.
        let maybe_to = find_underloaded_in_node(from, node_from, &lengths, mean, topology)
            .or_else(|| find_underloaded_any(from, &lengths, mean));

        let to = match maybe_to {
            Some(w) => w,
            None => break,
        };

        // Number of tasks to move: half the surplus, at least 1.
        let surplus = (lengths[from] - mean).floor() as usize;
        let n_move = (surplus / 2).max(1);

        // Check cost vs benefit: benefit = n_move (throughput units),
        // cost = migration_cost.
        let cost = migration_cost(from, to, topology, n_move);
        let benefit = n_move as f64;
        if cost > benefit {
            // Cross-NUMA migration not worth it; stop.
            break;
        }

        lengths[from] -= n_move as f64;
        lengths[to] += n_move as f64;
        plan.push((from, to, n_move));
    }

    plan
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Which NUMA node does worker `worker_id` belong to?
///
/// Worker IDs are assigned in node-major order:
///   worker i → core i → NUMA node (i / n_cores_per_node).
/// Falls back to node 0 for out-of-range worker IDs.
pub fn worker_numa_node(worker_id: usize, topology: &NumaTopology) -> usize {
    if topology.n_cores_per_node == 0 {
        return 0;
    }
    // Workers are mapped to cores in node-major order.
    let core_id = worker_id % topology.cores.len().max(1);
    node_of_core(topology, core_id)
}

/// Find the most underloaded worker on `node` (excluding `exclude`).
fn find_underloaded_in_node(
    exclude: usize,
    node: usize,
    lengths: &[f64],
    mean: f64,
    topology: &NumaTopology,
) -> Option<usize> {
    // Map logical worker IDs to cores in the given node.
    let cores_on_node = cores_in_node(topology, node);
    let n = lengths.len();

    // Worker IDs whose assigned core is on `node`.
    let candidates: Vec<usize> = (0..n)
        .filter(|&w| {
            if w == exclude {
                return false;
            }
            let core_id = w % topology.cores.len().max(1);
            cores_on_node.contains(&core_id)
        })
        .filter(|&w| lengths[w] < mean - 0.5)
        .collect();

    candidates.into_iter().min_by(|&a, &b| {
        lengths[a]
            .partial_cmp(&lengths[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Find the most underloaded worker across all nodes (excluding `exclude`).
fn find_underloaded_any(exclude: usize, lengths: &[f64], mean: f64) -> Option<usize> {
    (0..lengths.len())
        .filter(|&w| w != exclude && lengths[w] < mean - 0.5)
        .min_by(|&a, &b| {
            lengths[a]
                .partial_cmp(&lengths[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa_scheduler::types::NumaTopology;

    fn topo() -> NumaTopology {
        NumaTopology::from_config(2, 4)
    }

    #[test]
    fn test_load_imbalance_balanced() {
        let queues = vec![4, 4, 4, 4];
        let cov = load_imbalance(&queues);
        assert!(cov < 1e-9, "CoV should be ~0 for equal queues, got {}", cov);
    }

    #[test]
    fn test_load_imbalance_skewed() {
        let queues = vec![0, 0, 0, 100];
        let cov = load_imbalance(&queues);
        assert!(
            cov > 0.5,
            "CoV should be high for skewed queues, got {}",
            cov
        );
    }

    #[test]
    fn test_should_migrate_threshold() {
        assert!(should_migrate(0.6, 0.5));
        assert!(!should_migrate(0.4, 0.5));
        assert!(!should_migrate(0.5, 0.5)); // boundary: not strictly greater
    }

    #[test]
    fn test_migration_plan_intra_numa() {
        // 8 workers mapped 1:1 to 8 cores across 2 NUMA nodes.
        // Workers 0-3 → node 0; workers 4-7 → node 1.
        // Overload worker 0; underload worker 1 (same NUMA node 0).
        let t = topo();
        let queues = vec![20usize, 0, 2, 2, 2, 2, 2, 2];
        let plan = migration_plan(&queues, &t);
        // At least one migration should be proposed.
        assert!(!plan.is_empty(), "Expected at least one migration");
        // The first migration from worker 0 should go to a same-NUMA worker (1, 2, or 3).
        let (from, to, _n) = plan[0];
        assert_eq!(from, 0);
        assert!(to < 4, "Expected intra-NUMA target (0..4), got {}", to);
    }

    #[test]
    fn test_migration_cost_local_cheaper() {
        let t = topo();
        // Workers 0 and 1 are on node 0 (cores 0 and 1).
        let cost_local = migration_cost(0, 1, &t, 4);
        // Workers 0 and 4 are on different nodes.
        let cost_remote = migration_cost(0, 4, &t, 4);
        assert!(
            cost_local < cost_remote,
            "Local migration ({}) should be cheaper than remote ({})",
            cost_local,
            cost_remote
        );
    }

    #[test]
    fn test_migration_plan_no_op_when_balanced() {
        let t = topo();
        let queues = vec![5, 5, 5, 5, 5, 5, 5, 5];
        let plan = migration_plan(&queues, &t);
        assert!(plan.is_empty(), "No migrations needed for balanced queues");
    }

    #[test]
    fn test_worker_numa_node_mapping() {
        let t = topo(); // 2 nodes × 4 cores = 8 cores
                        // Workers 0-3 → cores 0-3 → node 0
        for w in 0..4 {
            assert_eq!(
                worker_numa_node(w, &t),
                0,
                "worker {} should be on node 0",
                w
            );
        }
        // Workers 4-7 → cores 4-7 → node 1
        for w in 4..8 {
            assert_eq!(
                worker_numa_node(w, &t),
                1,
                "worker {} should be on node 1",
                w
            );
        }
    }
}
