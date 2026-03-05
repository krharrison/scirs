//! Operator scheduling for computation graphs
//!
//! This module provides scheduling strategies for executing operations in
//! computation graphs, including topological ordering, memory-aware planning,
//! and parallel execution scheduling.
//!
//! # Sub-modules
//!
//! - [`topological`]: Forward, reverse, and memory-optimal topological sort
//! - [`memory_planner`]: Liveness analysis, in-place detection, memory reuse
//! - [`parallel_schedule`]: Critical path analysis, level parallelism, work-stealing

pub mod memory_planner;
pub mod parallel_schedule;
pub mod topological;

// Re-export key types for convenience
pub use memory_planner::{
    assign_memory_slots, build_memory_plan, detect_in_place, estimate_peak_memory,
    liveness_analysis, InPlaceCandidate, InPlaceReason, LivenessInterval, MemoryAssignment,
    MemoryPlan,
};
pub use parallel_schedule::{
    critical_path, level_decomposition, parallel_analysis, work_stealing_schedule, CriticalPath,
    ParallelAnalysis, ParallelLevel, ReadyTask, WorkStealingSchedule,
};
pub use topological::{
    compute_depth, forward_schedule, memory_optimal_schedule, reverse_schedule, validate_schedule,
    Schedule, ScheduleDirection, ScheduledOp,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AsGraph;
    use crate::tensor_ops as T;
    use crate::VariableEnvironment;

    /// Integration: run all scheduling strategies on the same graph and
    /// verify they all produce a valid schedule containing the same nodes.
    #[test]
    fn test_all_strategies_agree_on_node_set() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4, 4], ctx);
            let b = T::ones(&[4, 4], ctx);
            let c = a + b;
            let d = a * b;
            let e = c + d;
            let _ = e;

            let fwd = forward_schedule(ctx.as_graph());
            let rev = reverse_schedule(ctx.as_graph());
            let mem = memory_optimal_schedule(ctx.as_graph());

            assert_eq!(fwd.total_ops, rev.total_ops);
            assert_eq!(fwd.total_ops, mem.total_ops);

            // Forward schedule must be valid
            assert!(validate_schedule(ctx.as_graph(), &fwd).is_ok());
        });
    }

    /// Integration: memory plan + parallel analysis on a diamond graph.
    #[test]
    fn test_memory_plan_and_parallel_analysis() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[8, 8], ctx);
            let b = a + T::ones(&[8, 8], ctx);
            let c = a * T::ones(&[8, 8], ctx);
            let _ = b + c;

            let plan = build_memory_plan(ctx.as_graph());
            assert!(!plan.intervals.is_empty());
            assert!(plan.peak_memory > 0);

            let analysis = parallel_analysis(ctx.as_graph());
            assert!(analysis.total_work > 0);
            assert!(analysis.max_parallelism >= 1);
        });
    }

    /// Work-stealing schedule produces correct task count.
    #[test]
    fn test_work_stealing_distributes_all_tasks() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = T::ones(&[2], ctx);
            let c = a + b;
            let d = c * T::ones(&[2], ctx);
            let _ = d;

            let ws = work_stealing_schedule(ctx.as_graph(), 3);
            let distributed: usize = ws.worker_queues.iter().map(|q| q.len()).sum();
            assert_eq!(distributed, ws.total_tasks);
        });
    }
}
