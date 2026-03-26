//! Task dependency graph with topological scheduling, cycle detection, and critical-path analysis.
//!
//! This module provides a lightweight, metadata-only DAG for expressing tasks with explicit
//! data dependencies.  Unlike [`super::TaskGraph`] which carries typed compute closures,
//! [`DependencyGraph`] stores only scheduling metadata (name, priority, cost) so it can be
//! used for planning without executing.
//!
//! # Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`DependencyGraph`] | DAG container for task metadata and dependencies |
//! | [`DepTaskNode`] | Task node with name, priority, cost, and metadata |
//! | [`DependencyGraphConfig`] | Configuration for graph behaviour |
//! | [`TopologicalAlgorithm`] | Choice of Kahn vs DFS topological sort |
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::task_graph::dependency_graph::{DependencyGraph, DependencyGraphConfig};
//! use std::collections::HashSet;
//!
//! let mut g = DependencyGraph::new(DependencyGraphConfig::default());
//! let a = g.add_task("fetch", 1);
//! let b = g.add_task("process", 0);
//! let c = g.add_task("store", 0);
//! g.add_dependency(b, a).expect("b depends on a");
//! g.add_dependency(c, b).expect("c depends on b");
//!
//! let order = g.topological_sort().expect("acyclic");
//! assert_eq!(order.len(), 3);
//!
//! let completed: HashSet<u64> = [a].into();
//! assert!(g.is_ready(b, &completed));
//! assert!(!g.is_ready(c, &completed));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ─── TaskId ───────────────────────────────────────────────────────────────────

/// Opaque 64-bit identifier for tasks in a [`DependencyGraph`].
pub type TaskId = u64;

// ─── DepTaskNode ─────────────────────────────────────────────────────────────

/// A single node in a [`DependencyGraph`].
///
/// Stores task metadata without any compute closure.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DepTaskNode {
    /// Unique identifier for this task.
    pub id: TaskId,
    /// Human-readable name.
    pub name: String,
    /// Scheduling priority: higher value means execute first among ready tasks.
    pub priority: i32,
    /// Estimated execution cost (arbitrary units, used for critical-path analysis).
    pub estimated_cost: f64,
    /// Arbitrary key-value metadata for application-level annotations.
    pub metadata: HashMap<String, String>,
}

// ─── TopologicalAlgorithm ────────────────────────────────────────────────────

/// Choice of algorithm for topological sorting.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologicalAlgorithm {
    /// BFS-based algorithm by Kahn (1962).  Naturally detects cycles via in-degree counting.
    Kahn,
    /// DFS-based algorithm (Tarjan's SCC / finishing-time approach).
    DfsBased,
}

// ─── DependencyGraphConfig ───────────────────────────────────────────────────

/// Configuration for [`DependencyGraph`] behaviour.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DependencyGraphConfig {
    /// Whether to eagerly detect cycles when adding edges.  Default: `true`.
    pub enable_cycle_detection: bool,
    /// Maximum DFS depth before aborting to prevent stack overflow.  Default: 1000.
    pub max_depth: usize,
    /// Default algorithm for [`DependencyGraph::topological_sort`].
    pub topological_order: TopologicalAlgorithm,
}

impl Default for DependencyGraphConfig {
    fn default() -> Self {
        Self {
            enable_cycle_detection: true,
            max_depth: 1000,
            topological_order: TopologicalAlgorithm::Kahn,
        }
    }
}

// ─── DependencyGraph ─────────────────────────────────────────────────────────

/// A directed-acyclic graph (DAG) of tasks with metadata only.
///
/// Stores nodes, edges, and scheduling hints without holding compute closures.
/// Use this for planning, critical-path analysis, and dependency validation.
pub struct DependencyGraph {
    config: DependencyGraphConfig,
    /// All task nodes, keyed by their ID.
    nodes: HashMap<TaskId, DepTaskNode>,
    /// `edges[task]` = list of tasks that `task` depends on (predecessors).
    edges: HashMap<TaskId, Vec<TaskId>>,
    /// Reverse map: `rev_edges[dep]` = list of tasks that depend on `dep`.
    rev_edges: HashMap<TaskId, Vec<TaskId>>,
    next_id: TaskId,
}

impl DependencyGraph {
    /// Create an empty graph with the given configuration.
    pub fn new(config: DependencyGraphConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            rev_edges: HashMap::new(),
            next_id: 0,
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Add a task with `name` and `priority`.  Returns the new task's [`TaskId`].
    pub fn add_task(&mut self, name: &str, priority: i32) -> TaskId {
        self.add_task_with_cost(name, priority, 1.0)
    }

    /// Add a task with `name`, `priority`, and estimated `cost`.  Returns the task's [`TaskId`].
    pub fn add_task_with_cost(&mut self, name: &str, priority: i32, cost: f64) -> TaskId {
        let id = self.next_id;
        self.next_id += 1;
        let node = DepTaskNode {
            id,
            name: name.to_owned(),
            priority,
            estimated_cost: cost,
            metadata: HashMap::new(),
        };
        self.nodes.insert(id, node);
        self.edges.insert(id, Vec::new());
        self.rev_edges.insert(id, Vec::new());
        id
    }

    /// Declare that `task` depends on `dep` — `dep` must complete before `task`.
    ///
    /// When `enable_cycle_detection` is set, this checks for cycles before adding the edge.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidInput`] if either ID is unknown, if this is a self-loop,
    /// or if adding the edge would create a cycle (when cycle detection is enabled).
    pub fn add_dependency(&mut self, task: TaskId, dep: TaskId) -> CoreResult<()> {
        if !self.nodes.contains_key(&task) {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "add_dependency: task {task} not found"
            ))));
        }
        if !self.nodes.contains_key(&dep) {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "add_dependency: dep {dep} not found"
            ))));
        }
        if task == dep {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "add_dependency: self-loop on task {task}"
            ))));
        }
        // Cycle detection: would adding dep → task create a cycle?
        // That happens iff `dep` is already reachable from `task`.
        if self.config.enable_cycle_detection && self.is_reachable(dep, task) {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "add_dependency: cycle detected — dep {dep} is already reachable from task {task}"
            ))));
        }
        // Avoid duplicate edges.
        let deps = self.edges.entry(task).or_default();
        if !deps.contains(&dep) {
            deps.push(dep);
        }
        let rev = self.rev_edges.entry(dep).or_default();
        if !rev.contains(&task) {
            rev.push(task);
        }
        Ok(())
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Total number of task nodes.
    pub fn n_tasks(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of dependency edges.
    pub fn n_edges(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Return a reference to the task node for `id`, or `None` if not found.
    pub fn get_task(&self, id: TaskId) -> Option<&DepTaskNode> {
        self.nodes.get(&id)
    }

    /// Return the list of tasks that `id` directly depends on.
    pub fn dependencies(&self, id: TaskId) -> &[TaskId] {
        self.edges.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Compute the list of tasks that directly depend on `id` (reverse edges).
    pub fn dependents(&self, id: TaskId) -> Vec<TaskId> {
        self.rev_edges.get(&id).cloned().unwrap_or_default()
    }

    /// Return `true` if `id` is ready to run (all direct dependencies are in `completed`).
    pub fn is_ready(&self, id: TaskId, completed: &HashSet<TaskId>) -> bool {
        self.edges
            .get(&id)
            .map(|deps| deps.iter().all(|d| completed.contains(d)))
            .unwrap_or(true)
    }

    // ── Topological sort ──────────────────────────────────────────────────────

    /// Compute a topological ordering using the algorithm specified in the config.
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn topological_sort(&self) -> CoreResult<Vec<TaskId>> {
        match self.config.topological_order {
            TopologicalAlgorithm::Kahn => self.topological_sort_kahn(),
            TopologicalAlgorithm::DfsBased => self.topological_sort_dfs(),
        }
    }

    /// Kahn's algorithm (BFS-based topological sort).
    ///
    /// Returns `Err` if a cycle is detected.
    pub fn topological_sort_kahn(&self) -> CoreResult<Vec<TaskId>> {
        // in-degree = number of predecessors.
        let mut in_degree: HashMap<TaskId, usize> = self
            .nodes
            .keys()
            .map(|&id| (id, self.edges[&id].len()))
            .collect();

        // Priority queue implemented as sorted BTreeMap bucket — collect all zero-degree tasks.
        let mut ready: Vec<TaskId> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();
        // Sort by priority descending, then by id for determinism.
        ready.sort_unstable_by(|&a, &b| {
            let pa = self.nodes[&a].priority;
            let pb = self.nodes[&b].priority;
            pb.cmp(&pa).then(a.cmp(&b))
        });

        let mut order = Vec::with_capacity(self.nodes.len());
        while !ready.is_empty() {
            // Pop highest-priority task.
            let id = ready.remove(0);
            order.push(id);
            // Reduce in-degree of dependents.
            let new_ready: Vec<TaskId> = if let Some(children) = self.rev_edges.get(&id) {
                children
                    .iter()
                    .filter_map(|&child| {
                        let deg = in_degree.entry(child).or_insert(0);
                        if *deg > 0 {
                            *deg -= 1;
                        }
                        if *deg == 0 {
                            Some(child)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            // Insert new ready tasks in priority order.
            for nid in new_ready {
                let pos = ready.partition_point(|&x| {
                    let px = self.nodes[&x].priority;
                    let pn = self.nodes[&nid].priority;
                    px > pn || (px == pn && x < nid)
                });
                ready.insert(pos, nid);
            }
        }

        if order.len() != self.nodes.len() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "topological_sort: cycle detected in graph",
            )));
        }
        Ok(order)
    }

    /// DFS-based topological sort (finishing-time approach, Tarjan 1976).
    ///
    /// Follows the forward dependency direction (from predecessors to dependents) in DFS
    /// post-order.  Emitting tasks in post-order and reversing produces a valid topological
    /// ordering where all predecessors appear before their dependents.
    ///
    /// Returns `Err` if a cycle is detected.
    pub fn topological_sort_dfs(&self) -> CoreResult<Vec<TaskId>> {
        // Iterative DFS with explicit "return address" via a (node, child_index) stack.
        // This avoids recursion and correctly detects back edges.
        //
        // State per frame: (task_id, index_into_successors).
        // "Successors" in the forward-dependency direction are `rev_edges[node]`
        // (tasks that depend on `node`, i.e. tasks for which `node` is a prerequisite).

        // color: 0 = unvisited, 1 = in-stack (grey), 2 = finished (black)
        let mut color: HashMap<TaskId, u8> = self.nodes.keys().map(|&id| (id, 0u8)).collect();
        let mut result: Vec<TaskId> = Vec::with_capacity(self.nodes.len());

        // Deterministic start order (sorted by id).
        let mut all_ids: Vec<TaskId> = self.nodes.keys().cloned().collect();
        all_ids.sort_unstable();

        // Call stack: (node, iterator_index_into_successors)
        let mut call_stack: Vec<(TaskId, usize)> = Vec::new();

        for start in all_ids {
            if color[&start] != 0 {
                continue;
            }
            call_stack.push((start, 0));
            *color.entry(start).or_insert(0) = 1; // grey

            while let Some(frame) = call_stack.last_mut() {
                let (node, idx) = *frame;
                let successors: Vec<TaskId> =
                    self.rev_edges.get(&node).cloned().unwrap_or_default();

                if idx < successors.len() {
                    let child = successors[idx];
                    frame.1 += 1; // advance iterator
                    match color[&child] {
                        1 => {
                            // Back edge to a grey ancestor → cycle.
                            return Err(CoreError::InvalidInput(ErrorContext::new(
                                "topological_sort_dfs: cycle detected",
                            )));
                        }
                        0 => {
                            // Unvisited: push frame.
                            *color.entry(child).or_insert(0) = 1;
                            call_stack.push((child, 0));
                        }
                        _ => {} // Already finished (black) — skip.
                    }
                } else {
                    // All successors processed: finish node.
                    call_stack.pop();
                    *color.entry(node).or_insert(1) = 2;
                    result.push(node);
                }
            }
        }

        // DFS post-order on the forward graph gives reverse topological order.
        result.reverse();
        Ok(result)
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    /// Find all simple cycles in the graph.
    ///
    /// Returns each cycle as a `Vec<TaskId>` listing the nodes in cycle order.
    /// Returns an empty `Vec` if the graph is acyclic.
    pub fn find_cycles(&self) -> Vec<Vec<TaskId>> {
        // DFS with grey/black colouring; record back-edge chains as cycles.
        let mut color: HashMap<TaskId, u8> = self.nodes.keys().map(|&id| (id, 0u8)).collect();
        let mut cycles: Vec<Vec<TaskId>> = Vec::new();
        let mut stack: Vec<TaskId> = Vec::new();

        for &start in self.nodes.keys() {
            if color[&start] != 0 {
                continue;
            }
            self.dfs_find_cycles(start, &mut color, &mut stack, &mut cycles);
        }
        cycles
    }

    fn dfs_find_cycles(
        &self,
        node: TaskId,
        color: &mut HashMap<TaskId, u8>,
        stack: &mut Vec<TaskId>,
        cycles: &mut Vec<Vec<TaskId>>,
    ) {
        if stack.len() >= self.config.max_depth {
            return;
        }
        *color.entry(node).or_insert(0) = 1; // grey
        stack.push(node);

        let deps: Vec<TaskId> = self.edges.get(&node).cloned().unwrap_or_default();
        for dep in deps {
            match *color.entry(dep).or_insert(0) {
                1 => {
                    // Back edge: extract cycle.
                    if let Some(pos) = stack.iter().position(|&x| x == dep) {
                        let cycle: Vec<TaskId> = stack[pos..].to_vec();
                        cycles.push(cycle);
                    }
                }
                0 => self.dfs_find_cycles(dep, color, stack, cycles),
                _ => {}
            }
        }
        stack.pop();
        *color.entry(node).or_insert(1) = 2; // black
    }

    // ── Critical path ─────────────────────────────────────────────────────────

    /// Compute the critical path — the longest chain from a source to a sink,
    /// weighted by `estimated_cost`.
    ///
    /// Returns the ordered sequence of task IDs along that path.
    /// Returns an empty `Vec` if the graph is empty or contains a cycle.
    pub fn critical_path(&self) -> Vec<TaskId> {
        let order = match self.topological_sort() {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };
        // `dist[id]` = maximum cost to reach `id`.
        let mut dist: HashMap<TaskId, f64> = HashMap::new();
        let mut prev: HashMap<TaskId, Option<TaskId>> = HashMap::new();

        for &id in &order {
            let cost = self.nodes.get(&id).map(|n| n.estimated_cost).unwrap_or(1.0);
            let max_pred_dist = self
                .edges
                .get(&id)
                .map(|deps| {
                    deps.iter()
                        .filter_map(|d| dist.get(d).copied())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .unwrap_or(f64::NEG_INFINITY);
            let pred = if max_pred_dist.is_finite() {
                self.edges.get(&id).and_then(|deps| {
                    deps.iter()
                        .max_by(|&&a, &&b| {
                            dist.get(&a)
                                .copied()
                                .unwrap_or(f64::NEG_INFINITY)
                                .partial_cmp(&dist.get(&b).copied().unwrap_or(f64::NEG_INFINITY))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .copied()
                })
            } else {
                None
            };
            let d = if max_pred_dist.is_finite() {
                max_pred_dist + cost
            } else {
                cost
            };
            dist.insert(id, d);
            prev.insert(id, pred);
        }

        // Find sink with maximum distance.
        let sink = dist
            .iter()
            .max_by(|(_, &da), (_, &db)| da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, _)| id);

        let mut path = Vec::new();
        let mut current = sink;
        while let Some(id) = current {
            path.push(id);
            current = prev.get(&id).and_then(|opt| *opt);
        }
        path.reverse();
        path
    }

    // ── Execution layers ──────────────────────────────────────────────────────

    /// Group tasks into execution layers.
    ///
    /// Layer 0 contains all tasks with no dependencies.
    /// Layer `k` contains all tasks whose dependencies are in layers `< k`.
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn execution_layers(&self) -> CoreResult<Vec<Vec<TaskId>>> {
        // in-degree counts.
        let mut in_deg: HashMap<TaskId, usize> = self
            .nodes
            .keys()
            .map(|&id| (id, self.edges[&id].len()))
            .collect();

        let mut current_layer: Vec<TaskId> = in_deg
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();
        current_layer.sort_unstable();

        let mut layers: Vec<Vec<TaskId>> = Vec::new();
        let mut processed = 0usize;

        while !current_layer.is_empty() {
            layers.push(current_layer.clone());
            processed += current_layer.len();
            let mut next_layer: Vec<TaskId> = Vec::new();
            for id in &current_layer {
                if let Some(children) = self.rev_edges.get(id) {
                    for &child in children {
                        let deg = in_deg.entry(child).or_insert(0);
                        if *deg > 0 {
                            *deg -= 1;
                        }
                        if *deg == 0 {
                            next_layer.push(child);
                        }
                    }
                }
            }
            next_layer.sort_unstable();
            current_layer = next_layer;
        }

        if processed != self.nodes.len() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "execution_layers: cycle detected",
            )));
        }
        Ok(layers)
    }

    // ── Parallel schedule ─────────────────────────────────────────────────────

    /// Assign tasks to `n_workers` workers respecting dependencies.
    ///
    /// Returns `schedule[worker_id] = [task_id, ...]`.  Uses a greedy list-scheduling
    /// heuristic: assign the longest remaining task to the worker that becomes free first.
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn parallel_schedule(&self, n_workers: usize) -> CoreResult<Vec<Vec<TaskId>>> {
        let layers = self.execution_layers()?;
        let n_workers = n_workers.max(1);
        let mut schedule: Vec<Vec<TaskId>> = vec![Vec::new(); n_workers];

        // Assign tasks layer by layer in round-robin fashion within each layer.
        let mut worker = 0usize;
        for layer in &layers {
            // Sort layer by estimated cost descending (longest first).
            let mut sorted_layer = layer.clone();
            sorted_layer.sort_unstable_by(|&a, &b| {
                let ca = self.nodes.get(&a).map(|n| n.estimated_cost).unwrap_or(1.0);
                let cb = self.nodes.get(&b).map(|n| n.estimated_cost).unwrap_or(1.0);
                cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
            });
            for task_id in sorted_layer {
                schedule[worker % n_workers].push(task_id);
                worker += 1;
            }
        }
        Ok(schedule)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// BFS reachability: can `from` reach `target` by following edges?
    fn is_reachable(&self, from: TaskId, target: TaskId) -> bool {
        let mut visited: HashSet<TaskId> = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);
        while let Some(cur) = queue.pop_front() {
            if cur == target {
                return true;
            }
            if !visited.insert(cur) {
                continue;
            }
            if let Some(deps) = self.edges.get(&cur) {
                for &dep in deps {
                    if !visited.contains(&dep) {
                        queue.push_back(dep);
                    }
                }
            }
        }
        false
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DependencyGraphConfig {
        DependencyGraphConfig::default()
    }

    /// A → B → C chain.
    fn build_chain() -> (DependencyGraph, TaskId, TaskId, TaskId) {
        let mut g = DependencyGraph::new(make_config());
        let a = g.add_task("A", 0);
        let b = g.add_task("B", 0);
        let c = g.add_task("C", 0);
        g.add_dependency(b, a).expect("b depends on a");
        g.add_dependency(c, b).expect("c depends on b");
        (g, a, b, c)
    }

    #[test]
    fn test_add_task_and_dependency_no_error() {
        let mut g = DependencyGraph::new(make_config());
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        g.add_dependency(b, a).expect("valid DAG edge");
        assert_eq!(g.n_tasks(), 2);
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn test_topological_sort_chain() {
        let (g, a, b, c) = build_chain();
        let order = g.topological_sort().expect("acyclic");
        assert_eq!(order.len(), 3);
        let pos_a = order.iter().position(|&x| x == a).expect("a in order");
        let pos_b = order.iter().position(|&x| x == b).expect("b in order");
        let pos_c = order.iter().position(|&x| x == c).expect("c in order");
        assert!(pos_a < pos_b, "A must precede B");
        assert!(pos_b < pos_c, "B must precede C");
    }

    #[test]
    fn test_topological_sort_cycle_returns_err() {
        let mut g = DependencyGraph::new(DependencyGraphConfig {
            enable_cycle_detection: false,
            ..DependencyGraphConfig::default()
        });
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        // Manually insert cycle without triggering eager detection.
        g.edges.get_mut(&a).expect("a edges").push(b);
        g.edges.get_mut(&b).expect("b edges").push(a);
        assert!(
            g.topological_sort_kahn().is_err(),
            "cycle must be detected by Kahn"
        );
    }

    #[test]
    fn test_add_dependency_cycle_rejected() {
        let mut g = DependencyGraph::new(make_config());
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        g.add_dependency(b, a).expect("b → a");
        // Adding a → b would create a cycle.
        assert!(g.add_dependency(a, b).is_err(), "cycle must be rejected");
    }

    #[test]
    fn test_find_cycles_returns_cycle() {
        let mut g = DependencyGraph::new(DependencyGraphConfig {
            enable_cycle_detection: false,
            ..DependencyGraphConfig::default()
        });
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        g.edges.get_mut(&a).expect("a").push(b);
        g.edges.get_mut(&b).expect("b").push(a);
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty(), "should find at least one cycle");
    }

    #[test]
    fn test_execution_layers_independent_tasks_in_layer_0() {
        let mut g = DependencyGraph::new(make_config());
        g.add_task("x", 0);
        g.add_task("y", 0);
        g.add_task("z", 0);
        let layers = g.execution_layers().expect("acyclic");
        assert_eq!(layers.len(), 1, "all independent tasks in one layer");
        assert_eq!(layers[0].len(), 3);
    }

    #[test]
    fn test_execution_layers_chain() {
        let (g, _a, _b, _c) = build_chain();
        let layers = g.execution_layers().expect("acyclic");
        assert_eq!(layers.len(), 3, "chain has 3 layers");
        assert_eq!(layers[0].len(), 1); // A
        assert_eq!(layers[1].len(), 1); // B
        assert_eq!(layers[2].len(), 1); // C
    }

    #[test]
    fn test_critical_path_selects_longest_cost_path() {
        let mut g = DependencyGraph::new(make_config());
        // Two paths from source:
        // source → cheap (cost 1) → sink
        // source → expensive (cost 10) → sink
        let source = g.add_task_with_cost("source", 0, 1.0);
        let cheap = g.add_task_with_cost("cheap", 0, 1.0);
        let expensive = g.add_task_with_cost("expensive", 0, 10.0);
        let sink = g.add_task_with_cost("sink", 0, 1.0);
        g.add_dependency(cheap, source).expect("cheap dep");
        g.add_dependency(expensive, source).expect("expensive dep");
        g.add_dependency(sink, cheap).expect("sink dep cheap");
        g.add_dependency(sink, expensive)
            .expect("sink dep expensive");

        let path = g.critical_path();
        assert!(!path.is_empty(), "critical path should be non-empty");
        // The critical path must include `expensive` (highest cost intermediate).
        assert!(
            path.contains(&expensive),
            "critical path must go through 'expensive' node"
        );
    }

    #[test]
    fn test_parallel_schedule_all_tasks_covered() {
        let (g, _a, _b, _c) = build_chain();
        let schedule = g.parallel_schedule(2).expect("valid schedule");
        let all_tasks: HashSet<TaskId> = schedule.into_iter().flatten().collect();
        assert_eq!(all_tasks.len(), 3, "all tasks must be in schedule");
    }

    #[test]
    fn test_dependency_graph_config_default() {
        let cfg = DependencyGraphConfig::default();
        assert!(cfg.enable_cycle_detection);
        assert_eq!(cfg.max_depth, 1000);
        assert_eq!(cfg.topological_order, TopologicalAlgorithm::Kahn);
    }

    #[test]
    fn test_is_ready_task_with_all_deps_complete() {
        let mut g = DependencyGraph::new(make_config());
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        g.add_dependency(b, a).expect("b dep a");
        let completed: HashSet<TaskId> = [a].into();
        assert!(g.is_ready(b, &completed), "b is ready when a is complete");
        let empty: HashSet<TaskId> = HashSet::new();
        assert!(!g.is_ready(b, &empty), "b not ready when a is incomplete");
    }

    #[test]
    fn test_topological_sort_dfs_chain() {
        let (g, a, b, c) = build_chain();
        let order = g.topological_sort_dfs().expect("acyclic DFS");
        let pos_a = order.iter().position(|&x| x == a).expect("a in order");
        let pos_b = order.iter().position(|&x| x == b).expect("b in order");
        let pos_c = order.iter().position(|&x| x == c).expect("c in order");
        assert!(pos_a < pos_b, "DFS: A must precede B");
        assert!(pos_b < pos_c, "DFS: B must precede C");
    }

    #[test]
    fn test_dependencies_and_dependents() {
        let mut g = DependencyGraph::new(make_config());
        let a = g.add_task("a", 0);
        let b = g.add_task("b", 0);
        g.add_dependency(b, a).expect("b dep a");
        assert_eq!(g.dependencies(b), &[a]);
        assert_eq!(g.dependents(a), vec![b]);
    }

    #[test]
    fn test_get_task_metadata() {
        let mut g = DependencyGraph::new(make_config());
        let id = g.add_task_with_cost("my_task", 5, 42.0);
        let node = g.get_task(id).expect("task should exist");
        assert_eq!(node.name, "my_task");
        assert_eq!(node.priority, 5);
        assert!((node.estimated_cost - 42.0).abs() < f64::EPSILON);
    }
}
