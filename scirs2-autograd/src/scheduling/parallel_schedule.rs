//! Parallel execution scheduling for computation graphs
//!
//! Analyses the computation graph to identify operations that can run
//! concurrently, computes critical paths, and generates work-stealing
//! ready task lists for multi-threaded executors.

use crate::graph::{Graph, TensorID};
use crate::Float;
use std::collections::{HashMap, HashSet, VecDeque};

/// A level in the graph: all operations at the same depth that can run in parallel.
#[derive(Debug, Clone)]
pub struct ParallelLevel {
    /// Level index (0 = source nodes)
    pub level: usize,
    /// Node IDs at this level
    pub nodes: Vec<TensorID>,
    /// Maximum parallelism (number of nodes)
    pub width: usize,
}

/// Critical path information
#[derive(Debug, Clone)]
pub struct CriticalPath {
    /// Ordered nodes on the critical (longest) path from source to output
    pub path: Vec<TensorID>,
    /// Length of the critical path (number of operations)
    pub length: usize,
    /// Estimated total cost along the path (using unit cost per op)
    pub total_cost: f64,
}

/// A task ready for execution by a work-stealing scheduler
#[derive(Debug, Clone)]
pub struct ReadyTask {
    /// Node ID to execute
    pub node_id: TensorID,
    /// Priority (higher = should be scheduled sooner).
    /// Based on distance to output along the critical path.
    pub priority: usize,
    /// Operation name
    pub op_name: String,
    /// Input node IDs (dependencies)
    pub inputs: Vec<TensorID>,
    /// Number of downstream nodes that depend on this task
    pub num_dependents: usize,
}

/// Work-stealing task list: partitioned into per-worker queues
#[derive(Debug, Clone)]
pub struct WorkStealingSchedule {
    /// Number of worker threads
    pub num_workers: usize,
    /// Per-worker task queues (each sorted by priority, descending)
    pub worker_queues: Vec<Vec<ReadyTask>>,
    /// Total number of tasks
    pub total_tasks: usize,
    /// Maximum achievable parallelism (width of widest level)
    pub max_parallelism: usize,
    /// Critical path length (lower bound on total execution time)
    pub critical_path_length: usize,
    /// Estimated speedup = total_work / critical_path_length
    pub estimated_speedup: f64,
}

/// Parallel schedule analysis combining all parallel scheduling information
#[derive(Debug, Clone)]
pub struct ParallelAnalysis {
    /// Level-based decomposition
    pub levels: Vec<ParallelLevel>,
    /// Critical path
    pub critical_path: CriticalPath,
    /// Maximum parallelism (width of widest level)
    pub max_parallelism: usize,
    /// Average parallelism = total_ops / critical_path_length
    pub average_parallelism: f64,
    /// Work (total number of ops)
    pub total_work: usize,
    /// Span (critical path length)
    pub span: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Level-based parallelism
// ────────────────────────────────────────────────────────────────────────────

/// Compute the depth (longest path from any source) for every node.
fn compute_node_depths<F: Float>(graph: &Graph<F>) -> Vec<usize> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    let mut depth = vec![0usize; n];

    // Process in topo_rank order
    let mut order: Vec<TensorID> = (0..n).collect();
    order.sort_by_key(|&id| nodes[id].topo_rank);

    for &id in &order {
        for inc in &nodes[id].incoming_nodes {
            let pid = inc.id;
            if pid < n {
                let candidate = depth[pid] + 1;
                if candidate > depth[id] {
                    depth[id] = candidate;
                }
            }
        }
    }

    depth
}

/// Decompose a computation graph into parallel levels.
///
/// Nodes at the same depth (longest path from source) can execute in parallel
/// because they have no dependencies on each other at that depth.
pub fn level_decomposition<F: Float>(graph: &Graph<F>) -> Vec<ParallelLevel> {
    let depths = compute_node_depths(graph);
    let n = depths.len();
    if n == 0 {
        return Vec::new();
    }

    let max_depth = depths.iter().copied().max().unwrap_or(0);

    let mut levels: Vec<Vec<TensorID>> = vec![Vec::new(); max_depth + 1];
    for id in 0..n {
        levels[depths[id]].push(id);
    }

    levels
        .into_iter()
        .enumerate()
        .filter(|(_, nodes)| !nodes.is_empty())
        .map(|(level, nodes)| {
            let width = nodes.len();
            ParallelLevel {
                level,
                nodes,
                width,
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Critical path analysis
// ────────────────────────────────────────────────────────────────────────────

/// Compute the critical path (longest path from any source to any output).
///
/// Uses dynamic programming on the DAG: `dist[v] = max(dist[u] + cost(u,v))`
/// for all predecessors `u` of `v`. Then backtracks from the node with maximum
/// distance.
pub fn critical_path<F: Float>(graph: &Graph<F>) -> CriticalPath {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    if n == 0 {
        return CriticalPath {
            path: Vec::new(),
            length: 0,
            total_cost: 0.0,
        };
    }

    // Forward pass: compute longest distance from any source
    // Unit cost per operation for now
    let mut dist = vec![0usize; n];
    let mut predecessor = vec![None::<TensorID>; n];

    let mut order: Vec<TensorID> = (0..n).collect();
    order.sort_by_key(|&id| nodes[id].topo_rank);

    for &id in &order {
        for inc in &nodes[id].incoming_nodes {
            let pid = inc.id;
            if pid < n {
                let candidate = dist[pid] + 1;
                if candidate > dist[id] {
                    dist[id] = candidate;
                    predecessor[id] = Some(pid);
                }
            }
        }
    }

    // Find the node with maximum distance (the end of the critical path)
    let end_node = (0..n).max_by_key(|&id| dist[id]).unwrap_or(0);
    let length = dist[end_node];

    // Backtrack to reconstruct path
    let mut path = Vec::new();
    let mut current = Some(end_node);
    while let Some(nid) = current {
        path.push(nid);
        current = predecessor[nid];
    }
    path.reverse();

    CriticalPath {
        path,
        length,
        total_cost: length as f64,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Work-stealing schedule
// ────────────────────────────────────────────────────────────────────────────

/// Compute bottom-up priority for each node.
///
/// Priority = longest path from this node to any output.
/// Nodes on the critical path get the highest priority.
fn compute_priorities<F: Float>(graph: &Graph<F>) -> Vec<usize> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    let mut priority = vec![0usize; n];

    // Build children adjacency
    let mut children: Vec<Vec<TensorID>> = vec![Vec::new(); n];
    for node in nodes.iter() {
        for inc in &node.incoming_nodes {
            if inc.id < n {
                children[inc.id].push(node.id);
            }
        }
    }

    // Process in reverse topo order (output nodes first)
    let mut order: Vec<TensorID> = (0..n).collect();
    order.sort_by_key(|&id| std::cmp::Reverse(nodes[id].topo_rank));

    for &id in &order {
        for &child in &children[id] {
            let candidate = priority[child] + 1;
            if candidate > priority[id] {
                priority[id] = candidate;
            }
        }
    }

    priority
}

/// Generate a work-stealing schedule for a given number of workers.
///
/// Tasks are distributed round-robin by level, with critical-path tasks
/// assigned first to ensure they get priority.
pub fn work_stealing_schedule<F: Float>(
    graph: &Graph<F>,
    num_workers: usize,
) -> WorkStealingSchedule {
    let num_workers = num_workers.max(1);
    let nodes_ref = graph.node_set.borrow();
    let n = nodes_ref.len();

    if n == 0 {
        return WorkStealingSchedule {
            num_workers,
            worker_queues: vec![Vec::new(); num_workers],
            total_tasks: 0,
            max_parallelism: 0,
            critical_path_length: 0,
            estimated_speedup: 1.0,
        };
    }

    // Pre-collect node data to drop the borrow
    let node_data: Vec<(String, Vec<TensorID>)> = nodes_ref
        .iter()
        .map(|nd| {
            let op_name = nd
                .op
                .as_ref()
                .map(|o| o.name().to_owned())
                .unwrap_or_else(|| "source".to_owned());
            let inputs: Vec<TensorID> = nd.incoming_nodes.iter().map(|inc| inc.id).collect();
            (op_name, inputs)
        })
        .collect();

    // Build children adjacency for dependent count
    let mut children_count: Vec<usize> = vec![0; n];
    for nd in nodes_ref.iter() {
        for inc in &nd.incoming_nodes {
            if inc.id < n {
                children_count[inc.id] += 1;
            }
        }
    }

    drop(nodes_ref);

    let priorities = compute_priorities(graph);
    let levels = level_decomposition(graph);
    let cp = critical_path(graph);

    // Build tasks sorted by priority (descending)
    let mut all_tasks: Vec<ReadyTask> = (0..n)
        .map(|id| ReadyTask {
            node_id: id,
            priority: priorities[id],
            op_name: node_data[id].0.clone(),
            inputs: node_data[id].1.clone(),
            num_dependents: children_count[id],
        })
        .collect();
    all_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));

    // Distribute tasks round-robin to workers
    let mut worker_queues: Vec<Vec<ReadyTask>> = vec![Vec::new(); num_workers];
    for (i, task) in all_tasks.into_iter().enumerate() {
        worker_queues[i % num_workers].push(task);
    }

    let max_parallelism = levels.iter().map(|l| l.width).max().unwrap_or(1);
    let cp_len = cp.length;
    let speedup = if cp_len > 0 {
        n as f64 / cp_len as f64
    } else {
        1.0
    };

    WorkStealingSchedule {
        num_workers,
        worker_queues,
        total_tasks: n,
        max_parallelism,
        critical_path_length: cp_len,
        estimated_speedup: speedup,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Full parallel analysis
// ────────────────────────────────────────────────────────────────────────────

/// Perform full parallel analysis on a computation graph.
pub fn parallel_analysis<F: Float>(graph: &Graph<F>) -> ParallelAnalysis {
    let levels = level_decomposition(graph);
    let cp = critical_path(graph);
    let total_work: usize = levels.iter().map(|l| l.width).sum();
    let max_par = levels.iter().map(|l| l.width).max().unwrap_or(0);
    let span = cp.length;
    let avg_par = if span > 0 {
        total_work as f64 / span as f64
    } else if total_work > 0 {
        total_work as f64
    } else {
        0.0
    };

    ParallelAnalysis {
        levels,
        critical_path: cp,
        max_parallelism: max_par,
        average_parallelism: avg_par,
        total_work,
        span,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AsGraph;
    use crate::tensor_ops as T;
    use crate::VariableEnvironment;

    #[test]
    fn test_level_decomposition_linear() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = a + T::ones(&[2], ctx);
            let _ = b * T::ones(&[2], ctx);

            let levels = level_decomposition(ctx.as_graph());
            assert!(!levels.is_empty());
            // Source nodes should be at level 0
            assert_eq!(levels[0].level, 0);
        });
    }

    #[test]
    fn test_level_decomposition_wide() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            // Three independent source nodes
            let a = T::zeros(&[2], ctx);
            let b = T::ones(&[2], ctx);
            let c = T::zeros(&[2], ctx);
            // All consumed at the same level
            let d = a + b;
            let _ = d + c;

            let levels = level_decomposition(ctx.as_graph());
            // Level 0 should have multiple nodes (all sources)
            assert!(levels[0].width >= 2, "Expected wide first level");
        });
    }

    #[test]
    fn test_critical_path() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = a + T::ones(&[2], ctx);
            let c = b * T::ones(&[2], ctx);
            let _ = c + T::ones(&[2], ctx);

            let cp = critical_path(ctx.as_graph());
            assert!(cp.length >= 1, "Critical path should have length >= 1");
            assert!(!cp.path.is_empty());
        });
    }

    #[test]
    fn test_critical_path_empty_graph() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let cp = critical_path(ctx.as_graph());
            assert_eq!(cp.length, 0);
            assert!(cp.path.is_empty());
        });
    }

    #[test]
    fn test_work_stealing_schedule() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[3], ctx);
            let b = T::ones(&[3], ctx);
            let c = a + b;
            let d = a * b;
            let _ = c + d;

            let ws = work_stealing_schedule(ctx.as_graph(), 4);
            assert_eq!(ws.num_workers, 4);
            assert!(ws.total_tasks > 0);
            assert!(ws.max_parallelism >= 1);

            // All tasks should be distributed
            let total_distributed: usize = ws.worker_queues.iter().map(|q| q.len()).sum();
            assert_eq!(total_distributed, ws.total_tasks);
        });
    }

    #[test]
    fn test_work_stealing_single_worker() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let _ = a + T::ones(&[2], ctx);

            let ws = work_stealing_schedule(ctx.as_graph(), 1);
            assert_eq!(ws.num_workers, 1);
            assert_eq!(ws.worker_queues.len(), 1);
            assert_eq!(ws.worker_queues[0].len(), ws.total_tasks);
        });
    }

    #[test]
    fn test_parallel_analysis() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4], ctx);
            let b = T::ones(&[4], ctx);
            let c = a + b;
            let d = a * b;
            let _ = c + d;

            let analysis = parallel_analysis(ctx.as_graph());
            assert!(analysis.total_work > 0);
            assert!(analysis.max_parallelism >= 1);
            assert!(analysis.average_parallelism > 0.0);
        });
    }

    #[test]
    fn test_parallel_analysis_empty() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let analysis = parallel_analysis(ctx.as_graph());
            assert_eq!(analysis.total_work, 0);
            assert_eq!(analysis.max_parallelism, 0);
        });
    }

    #[test]
    fn test_task_priorities() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = a + T::ones(&[2], ctx);
            let _ = b * T::ones(&[2], ctx);

            let priorities = compute_priorities(ctx.as_graph());
            // Source nodes feeding into deep chains should have higher priority
            // than leaf output nodes
            assert!(!priorities.is_empty());
        });
    }
}
