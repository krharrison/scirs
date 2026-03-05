//! DAG-based task graph executor
//!
//! This module provides a directed acyclic graph (DAG) based task dependency
//! graph with automatic parallelism. Tasks are added with named dependencies,
//! and execution proceeds in topological order with maximal concurrency.
//!
//! ## Features
//!
//! - **Automatic parallelism**: Independent tasks run concurrently
//! - **Cycle detection**: Adding a task that would create a cycle is rejected
//! - **Topological ordering**: Tasks execute in valid dependency order
//! - **Result collection**: Each task produces a result accessible by name
//! - **Error propagation**: Failures in dependencies prevent dependent tasks
//!   from executing
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::distributed::task_graph::TaskGraph;
//!
//! let mut graph = TaskGraph::<i32>::new();
//! graph.add_task("load_data", &[], |_results| Ok(42)).expect("add failed");
//! graph.add_task("process", &["load_data"], |results| {
//!     let data = results.get("load_data").copied().unwrap_or(0);
//!     Ok(data * 2)
//! }).expect("add failed");
//! graph.add_task("save", &["process"], |results| {
//!     let processed = results.get("process").copied().unwrap_or(0);
//!     Ok(processed + 1)
//! }).expect("add failed");
//!
//! let results = graph.execute().expect("execution failed");
//! assert_eq!(results.get("load_data"), Some(&42));
//! assert_eq!(results.get("process"), Some(&84));
//! assert_eq!(results.get("save"), Some(&85));
//! ```

use crate::error::{CoreError, ErrorContext, ErrorLocation};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;

/// Result type alias for task graph operations.
pub type TaskGraphResult<T> = Result<T, CoreError>;

/// A task closure that receives completed results and produces a value.
///
/// The `HashMap<String, T>` contains the results of all completed dependency
/// tasks, keyed by task name.
type TaskFn<T> = Box<dyn FnOnce(&HashMap<String, T>) -> TaskGraphResult<T> + Send>;

/// Internal representation of a task node in the dependency graph.
struct TaskNode<T: Send + 'static> {
    /// Unique task name.
    name: String,
    /// Names of tasks this node depends on.
    dependencies: Vec<String>,
    /// The closure to execute.
    func: Option<TaskFn<T>>,
}

/// A directed acyclic graph of tasks with automatic parallel execution.
///
/// Tasks are added via [`add_task`](TaskGraph::add_task) and executed via
/// [`execute`](TaskGraph::execute). Independent tasks (those whose
/// dependencies are all satisfied) run concurrently on OS threads.
///
/// `T` must be `Clone + Send + 'static` so that results can be shared
/// between threads and stored in the result map.
pub struct TaskGraph<T: Clone + Send + 'static> {
    /// All registered task nodes, keyed by name.
    nodes: HashMap<String, TaskNode<T>>,
    /// Insertion order for deterministic iteration.
    insertion_order: Vec<String>,
}

impl<T: Clone + Send + 'static> TaskGraph<T> {
    /// Create an empty task graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    /// Number of tasks in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the graph contains no tasks.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns `true` if the graph contains a task with the given name.
    pub fn contains_task(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Get the dependency names for a task.
    pub fn dependencies(&self, name: &str) -> Option<&[String]> {
        self.nodes.get(name).map(|n| n.dependencies.as_slice())
    }

    /// Add a task to the graph.
    ///
    /// - `name`: unique identifier for the task.
    /// - `deps`: names of tasks that must complete before this one.
    /// - `func`: closure receiving a map of completed dependency results.
    ///
    /// # Errors
    ///
    /// - [`CoreError::ValueError`] if `name` is already taken.
    /// - [`CoreError::ValueError`] if any dependency name is not yet registered.
    /// - [`CoreError::ComputationError`] if adding this task would create a cycle.
    pub fn add_task<F>(&mut self, name: &str, deps: &[&str], func: F) -> TaskGraphResult<()>
    where
        F: FnOnce(&HashMap<String, T>) -> TaskGraphResult<T> + Send + 'static,
    {
        // Check for duplicate name
        if self.nodes.contains_key(name) {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!("Task '{name}' already exists in the graph"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check that all dependencies exist
        for dep in deps {
            if !self.nodes.contains_key(*dep) {
                return Err(CoreError::ValueError(
                    ErrorContext::new(format!(
                        "Dependency '{dep}' for task '{name}' does not exist in the graph"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }

        let dep_names: Vec<String> = deps.iter().map(|d| d.to_string()).collect();

        // Temporarily insert to check for cycles
        self.nodes.insert(
            name.to_string(),
            TaskNode {
                name: name.to_string(),
                dependencies: dep_names.clone(),
                func: None, // placeholder
            },
        );

        if self.has_cycle() {
            // Remove the node we just inserted
            self.nodes.remove(name);
            return Err(CoreError::ComputationError(
                ErrorContext::new(format!(
                    "Adding task '{name}' would create a cycle in the dependency graph"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Replace placeholder with real func
        if let Some(node) = self.nodes.get_mut(name) {
            node.func = Some(Box::new(func));
        }

        self.insertion_order.push(name.to_string());
        Ok(())
    }

    /// Detect whether the graph contains a cycle using Kahn's algorithm.
    fn has_cycle(&self) -> bool {
        let topo = self.topological_sort();
        // If topological sort returns fewer nodes than we have, there is a cycle.
        topo.len() != self.nodes.len()
    }

    /// Compute a topological ordering of the tasks using Kahn's algorithm.
    ///
    /// Returns a `Vec<String>` of task names in valid execution order.
    /// If the graph has a cycle, the returned vector will be shorter than
    /// the number of nodes.
    fn topological_sort(&self) -> Vec<String> {
        // Build in-degree map
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for (name, node) in &self.nodes {
            in_degree.entry(name.as_str()).or_insert(0);
            for dep in &node.dependencies {
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(name.as_str());
                *in_degree.entry(name.as_str()).or_insert(0) += 1;
            }
        }

        // Seed queue with zero-indegree nodes
        let mut queue: VecDeque<&str> = VecDeque::new();
        // Use insertion_order for deterministic iteration when possible
        for name in &self.insertion_order {
            if let Some(&deg) = in_degree.get(name.as_str()) {
                if deg == 0 {
                    queue.push_back(name.as_str());
                }
            }
        }
        // Also handle nodes not in insertion_order (shouldn't happen, but safe)
        for name in self.nodes.keys() {
            if !self.insertion_order.contains(name) {
                if let Some(&deg) = in_degree.get(name.as_str()) {
                    if deg == 0 && !queue.contains(&name.as_str()) {
                        queue.push_back(name.as_str());
                    }
                }
            }
        }

        let mut order: Vec<String> = Vec::with_capacity(self.nodes.len());

        while let Some(current) = queue.pop_front() {
            order.push(current.to_string());
            if let Some(deps) = dependents.get(current) {
                for &dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        order
    }

    /// Compute the level (longest path from any root) for each task.
    ///
    /// Tasks at the same level have no dependencies on each other and can
    /// run concurrently.
    fn compute_levels(&self) -> Vec<Vec<String>> {
        let topo = self.topological_sort();
        let mut level_of: HashMap<String, usize> = HashMap::new();

        for name in &topo {
            let node = match self.nodes.get(name) {
                Some(n) => n,
                None => continue,
            };
            let max_dep_level = node
                .dependencies
                .iter()
                .filter_map(|d| level_of.get(d))
                .copied()
                .max()
                .map(|l| l + 1)
                .unwrap_or(0);
            level_of.insert(name.clone(), max_dep_level);
        }

        // Group by level
        let max_level = level_of.values().copied().max().unwrap_or(0);
        let mut levels: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];
        for (name, level) in &level_of {
            levels[*level].push(name.clone());
        }

        levels
    }

    /// Execute all tasks in the graph, respecting dependencies.
    ///
    /// Tasks at the same dependency level run concurrently on OS threads.
    /// Returns a map from task name to result.
    ///
    /// # Errors
    ///
    /// - If a task closure returns an error, execution continues for
    ///   independent tasks but dependent tasks will not run.
    /// - Returns the first error encountered.
    pub fn execute(mut self) -> TaskGraphResult<HashMap<String, T>> {
        if self.nodes.is_empty() {
            return Ok(HashMap::new());
        }

        let levels = self.compute_levels();
        let results: Arc<Mutex<HashMap<String, T>>> = Arc::new(Mutex::new(HashMap::new()));
        let errors: Arc<Mutex<Vec<(String, CoreError)>>> = Arc::new(Mutex::new(Vec::new()));

        // Track which tasks failed (or had failed dependencies)
        let failed_tasks: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

        for level_tasks in &levels {
            if level_tasks.is_empty() {
                continue;
            }

            if level_tasks.len() == 1 {
                // Single task: run on current thread
                let task_name = &level_tasks[0];

                // Check if any dependency failed
                let dep_failed = {
                    let ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    let node = self.nodes.get(task_name);
                    node.map(|n| n.dependencies.iter().any(|d| ft.contains(d)))
                        .unwrap_or(false)
                };

                if dep_failed {
                    let mut ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    ft.insert(task_name.clone());
                    continue;
                }

                if let Some(node) = self.nodes.get_mut(task_name) {
                    if let Some(func) = node.func.take() {
                        let res_snapshot = {
                            let r = results.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock results".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            r.clone()
                        };
                        match func(&res_snapshot) {
                            Ok(val) => {
                                let mut r = results.lock().map_err(|_| {
                                    CoreError::MutexError(
                                        ErrorContext::new("Failed to lock results".to_string())
                                            .with_location(ErrorLocation::new(file!(), line!())),
                                    )
                                })?;
                                r.insert(task_name.clone(), val);
                            }
                            Err(e) => {
                                let mut ft = failed_tasks.lock().map_err(|_| {
                                    CoreError::MutexError(
                                        ErrorContext::new(
                                            "Failed to lock failed_tasks".to_string(),
                                        )
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                    )
                                })?;
                                ft.insert(task_name.clone());
                                let mut errs = errors.lock().map_err(|_| {
                                    CoreError::MutexError(
                                        ErrorContext::new("Failed to lock errors".to_string())
                                            .with_location(ErrorLocation::new(file!(), line!())),
                                    )
                                })?;
                                errs.push((task_name.clone(), e));
                            }
                        }
                    }
                }
            } else {
                // Multiple tasks: run concurrently
                // Extract task closures
                let mut task_closures: Vec<(String, TaskFn<T>, Vec<String>)> = Vec::new();
                for task_name in level_tasks {
                    if let Some(node) = self.nodes.get_mut(task_name) {
                        if let Some(func) = node.func.take() {
                            task_closures.push((
                                task_name.clone(),
                                func,
                                node.dependencies.clone(),
                            ));
                        }
                    }
                }

                // Take a snapshot of current results
                let res_snapshot = {
                    let r = results.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock results".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    r.clone()
                };

                let failed_snapshot: HashSet<String> = {
                    let ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    ft.clone()
                };

                // Spawn threads for concurrent execution
                let mut handles: Vec<(String, thread::JoinHandle<Result<T, CoreError>>)> =
                    Vec::new();
                let mut skipped: Vec<String> = Vec::new();

                for (task_name, func, deps) in task_closures {
                    let dep_failed = deps.iter().any(|d| failed_snapshot.contains(d));
                    if dep_failed {
                        skipped.push(task_name);
                        continue;
                    }

                    let snapshot = res_snapshot.clone();
                    let handle = thread::spawn(move || func(&snapshot));
                    handles.push((task_name, handle));
                }

                // Mark skipped tasks as failed
                {
                    let mut ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    for s in skipped {
                        ft.insert(s);
                    }
                }

                // Collect results
                for (task_name, handle) in handles {
                    match handle.join() {
                        Ok(Ok(val)) => {
                            let mut r = results.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock results".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            r.insert(task_name, val);
                        }
                        Ok(Err(e)) => {
                            let mut ft = failed_tasks.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock failed_tasks".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            ft.insert(task_name.clone());
                            let mut errs = errors.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock errors".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            errs.push((task_name, e));
                        }
                        Err(_panic) => {
                            let mut ft = failed_tasks.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock failed_tasks".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            ft.insert(task_name.clone());
                            let mut errs = errors.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock errors".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            errs.push((
                                task_name,
                                CoreError::ThreadError(
                                    ErrorContext::new("Task thread panicked".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                ),
                            ));
                        }
                    }
                }
            }
        }

        // Check for errors
        let errs = errors.lock().map_err(|_| {
            CoreError::MutexError(
                ErrorContext::new("Failed to lock errors".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        if let Some((task_name, err)) = errs.first() {
            return Err(CoreError::ComputationError(
                ErrorContext::new(format!("Task '{task_name}' failed: {err}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let final_results = results.lock().map_err(|_| {
            CoreError::MutexError(
                ErrorContext::new("Failed to lock results".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(final_results.clone())
    }

    /// Execute the graph and return only successfully completed results,
    /// ignoring failures (partial execution).
    ///
    /// Unlike [`execute`](TaskGraph::execute), this never returns an error
    /// for task failures. Tasks with unsatisfied dependencies are silently
    /// skipped.
    pub fn execute_partial(self) -> TaskGraphResult<HashMap<String, T>> {
        // Re-use the same logic but capture errors differently
        // We implement this by converting the error result to partial results
        let levels = self.compute_levels();
        let mut all_nodes = self.nodes;
        let results: Arc<Mutex<HashMap<String, T>>> = Arc::new(Mutex::new(HashMap::new()));
        let failed_tasks: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

        for level_tasks in &levels {
            for task_name in level_tasks {
                let dep_failed = {
                    let ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    let node = all_nodes.get(task_name);
                    node.map(|n| n.dependencies.iter().any(|d| ft.contains(d)))
                        .unwrap_or(false)
                };

                if dep_failed {
                    let mut ft = failed_tasks.lock().map_err(|_| {
                        CoreError::MutexError(
                            ErrorContext::new("Failed to lock failed_tasks".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    ft.insert(task_name.clone());
                    continue;
                }

                if let Some(node) = all_nodes.get_mut(task_name) {
                    if let Some(func) = node.func.take() {
                        let res_snapshot = {
                            let r = results.lock().map_err(|_| {
                                CoreError::MutexError(
                                    ErrorContext::new("Failed to lock results".to_string())
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                )
                            })?;
                            r.clone()
                        };
                        match func(&res_snapshot) {
                            Ok(val) => {
                                let mut r = results.lock().map_err(|_| {
                                    CoreError::MutexError(
                                        ErrorContext::new("Failed to lock results".to_string())
                                            .with_location(ErrorLocation::new(file!(), line!())),
                                    )
                                })?;
                                r.insert(task_name.clone(), val);
                            }
                            Err(_) => {
                                let mut ft = failed_tasks.lock().map_err(|_| {
                                    CoreError::MutexError(
                                        ErrorContext::new(
                                            "Failed to lock failed_tasks".to_string(),
                                        )
                                        .with_location(ErrorLocation::new(file!(), line!())),
                                    )
                                })?;
                                ft.insert(task_name.clone());
                            }
                        }
                    }
                }
            }
        }

        let final_results = results.lock().map_err(|_| {
            CoreError::MutexError(
                ErrorContext::new("Failed to lock results".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(final_results.clone())
    }

    /// Return a topological ordering of task names.
    ///
    /// Useful for debugging or inspection.
    pub fn execution_order(&self) -> Vec<String> {
        self.topological_sort()
    }

    /// Return task names grouped by execution level.
    ///
    /// Tasks within the same level can execute concurrently.
    pub fn execution_levels(&self) -> Vec<Vec<String>> {
        self.compute_levels()
    }
}

impl<T: Clone + Send + 'static> Default for TaskGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = TaskGraph::<i32>::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        let results = graph.execute().expect("empty graph should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_task() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("only", &[], |_| Ok(99)).expect("add failed");
        assert_eq!(graph.len(), 1);
        assert!(graph.contains_task("only"));

        let results = graph.execute().expect("execute failed");
        assert_eq!(results.get("only"), Some(&99));
    }

    #[test]
    fn test_linear_chain() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("a", &[], |_| Ok(1)).expect("add a");
        graph
            .add_task("b", &["a"], |r| Ok(r.get("a").copied().unwrap_or(0) + 10))
            .expect("add b");
        graph
            .add_task("c", &["b"], |r| Ok(r.get("b").copied().unwrap_or(0) + 100))
            .expect("add c");

        let results = graph.execute().expect("execute failed");
        assert_eq!(results.get("a"), Some(&1));
        assert_eq!(results.get("b"), Some(&11));
        assert_eq!(results.get("c"), Some(&111));
    }

    #[test]
    fn test_diamond_dependency() {
        // A -> B, A -> C, B -> D, C -> D
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("a", &[], |_| Ok(1)).expect("add a");
        graph
            .add_task("b", &["a"], |r| Ok(r.get("a").copied().unwrap_or(0) * 2))
            .expect("add b");
        graph
            .add_task("c", &["a"], |r| Ok(r.get("a").copied().unwrap_or(0) * 3))
            .expect("add c");
        graph
            .add_task("d", &["b", "c"], |r| {
                let b = r.get("b").copied().unwrap_or(0);
                let c = r.get("c").copied().unwrap_or(0);
                Ok(b + c)
            })
            .expect("add d");

        let results = graph.execute().expect("execute failed");
        assert_eq!(results.get("a"), Some(&1));
        assert_eq!(results.get("b"), Some(&2));
        assert_eq!(results.get("c"), Some(&3));
        assert_eq!(results.get("d"), Some(&5));
    }

    #[test]
    fn test_parallel_independent_tasks() {
        let mut graph = TaskGraph::<String>::new();
        for i in 0..8 {
            let name = format!("task_{i}");
            graph
                .add_task(&name, &[], move |_| Ok(format!("result_{i}")))
                .expect("add failed");
        }

        let levels = graph.execution_levels();
        // All tasks should be at level 0
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 8);

        let results = graph.execute().expect("execute failed");
        assert_eq!(results.len(), 8);
        for i in 0..8 {
            assert_eq!(
                results.get(&format!("task_{i}")),
                Some(&format!("result_{i}"))
            );
        }
    }

    #[test]
    fn test_duplicate_task_name_rejected() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("x", &[], |_| Ok(1)).expect("add x");
        let err = graph.add_task("x", &[], |_| Ok(2));
        assert!(err.is_err());
    }

    #[test]
    fn test_missing_dependency_rejected() {
        let mut graph = TaskGraph::<i32>::new();
        let err = graph.add_task("x", &["nonexistent"], |_| Ok(1));
        assert!(err.is_err());
    }

    #[test]
    fn test_cycle_detection() {
        // We can't directly create a cycle because deps must exist first,
        // but we can test the internal cycle detection
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("a", &[], |_| Ok(1)).expect("add a");
        graph.add_task("b", &["a"], |_| Ok(2)).expect("add b");
        // Trying to make a depend on b would fail because a is already added
        // and we'd need to re-add it. The API prevents cycles by design.
        // Let's verify the graph is acyclic
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_task_failure_propagation() {
        let mut graph = TaskGraph::<i32>::new();
        graph
            .add_task("fail", &[], |_| {
                Err(CoreError::ComputationError(
                    ErrorContext::new("intentional failure".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            })
            .expect("add fail");
        graph
            .add_task("downstream", &["fail"], |_| Ok(42))
            .expect("add downstream");

        let result = graph.execute();
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_execution() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("ok", &[], |_| Ok(10)).expect("add ok");
        graph
            .add_task("fail", &[], |_| {
                Err(CoreError::ComputationError(
                    ErrorContext::new("boom".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            })
            .expect("add fail");
        graph
            .add_task("depends_on_fail", &["fail"], |_| Ok(20))
            .expect("add depends");

        let results = graph.execute_partial().expect("partial should not error");
        assert_eq!(results.get("ok"), Some(&10));
        assert!(!results.contains_key("fail"));
        assert!(!results.contains_key("depends_on_fail"));
    }

    #[test]
    fn test_execution_order() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("a", &[], |_| Ok(1)).expect("add a");
        graph.add_task("b", &["a"], |_| Ok(2)).expect("add b");
        graph.add_task("c", &["b"], |_| Ok(3)).expect("add c");

        let order = graph.execution_order();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_execution_levels_structure() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("a", &[], |_| Ok(1)).expect("add a");
        graph.add_task("b", &[], |_| Ok(2)).expect("add b");
        graph.add_task("c", &["a", "b"], |_| Ok(3)).expect("add c");

        let levels = graph.execution_levels();
        assert_eq!(levels.len(), 2);
        // Level 0: a, b (independent)
        assert_eq!(levels[0].len(), 2);
        assert!(levels[0].contains(&"a".to_string()));
        assert!(levels[0].contains(&"b".to_string()));
        // Level 1: c (depends on a and b)
        assert_eq!(levels[1].len(), 1);
        assert!(levels[1].contains(&"c".to_string()));
    }

    #[test]
    fn test_wide_fan_in() {
        let mut graph = TaskGraph::<i32>::new();
        let n = 16;
        let mut dep_names: Vec<String> = Vec::new();
        for i in 0..n {
            let name = format!("src_{i}");
            graph
                .add_task(&name, &[], move |_| Ok(i as i32))
                .expect("add src");
            dep_names.push(name);
        }
        let dep_refs: Vec<&str> = dep_names.iter().map(|s| s.as_str()).collect();
        graph
            .add_task("sink", &dep_refs, |r| Ok(r.values().sum::<i32>()))
            .expect("add sink");

        let results = graph.execute().expect("execute failed");
        let expected_sum: i32 = (0..n as i32).sum();
        assert_eq!(results.get("sink"), Some(&expected_sum));
    }

    #[test]
    fn test_dependencies_accessor() {
        let mut graph = TaskGraph::<i32>::new();
        graph.add_task("root", &[], |_| Ok(0)).expect("add root");
        graph
            .add_task("child", &["root"], |_| Ok(1))
            .expect("add child");

        let deps = graph.dependencies("child").expect("should exist");
        assert_eq!(deps, &["root".to_string()]);
        assert!(graph.dependencies("nonexistent").is_none());
    }
}
