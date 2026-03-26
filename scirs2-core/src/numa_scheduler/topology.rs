/// NUMA topology detection and query helpers.
///
/// Because we target pure Rust without OS-specific libc calls, all topology
/// information is *simulated*.  In production you would swap
/// [`detect_topology`] for a platform-specific implementation that reads
/// `/sys/devices/system/node/` (Linux) or calls `GetNumaNodeProcessorMask`
/// (Windows).  The rest of the API is independent of how the topology was
/// obtained.
use crate::numa_scheduler::types::{CoreInfo, NumaTopology};

// ─── Public topology queries ─────────────────────────────────────────────────

/// Build a default simulated topology: 2 NUMA nodes × 4 cores each.
///
/// This is the value returned when no OS information is available.  Callers
/// can override with [`NumaTopology::from_config`].
pub fn detect_topology() -> NumaTopology {
    NumaTopology::from_config(2, 4)
}

// ─── Distance matrix ──────────────────────────────────────────────────────────

/// Communication cost between two NUMA nodes.
///
/// * Same node  → 10 (local memory access).
/// * Different nodes → 20 (remote NUMA hop, single hop assumed).
///
/// For topologies with more than two nodes the cost could be a function of
/// the hop count; we keep it simple here.
pub fn distance(node_a: usize, node_b: usize) -> usize {
    if node_a == node_b {
        10
    } else {
        20
    }
}

/// Return all NUMA node IDs sorted by distance from `from`, nearest first.
///
/// Ties (nodes equidistant from `from`) are broken by node index.
pub fn nearest_numa_nodes(topology: &NumaTopology, from: usize) -> Vec<usize> {
    let mut nodes: Vec<usize> = (0..topology.n_nodes).collect();
    nodes.sort_by_key(|&n| (distance(from, n), n));
    nodes
}

// ─── Core affinity mapping ────────────────────────────────────────────────────

/// Return the core IDs that belong to the given NUMA node.
///
/// Returns an empty vector if `node` is out of range.
pub fn cores_in_node(topology: &NumaTopology, node: usize) -> Vec<usize> {
    topology
        .cores
        .iter()
        .filter(|c| c.numa_node == node)
        .map(|c| c.core_id)
        .collect()
}

/// Return the NUMA node that contains `core`.
///
/// Returns `0` (the first node) if `core` is not found — callers should
/// validate core IDs before calling this.
pub fn node_of_core(topology: &NumaTopology, core: usize) -> usize {
    topology
        .cores
        .iter()
        .find(|c| c.core_id == core)
        .map(|c| c.numa_node)
        .unwrap_or(0)
}

// ─── Cache topology ───────────────────────────────────────────────────────────

/// Return all cores that share the L3 cache with `core`.
///
/// In our simulated model, cores on the same NUMA node share an L3 cache.
/// The result always includes `core` itself.
pub fn shared_l3_cores(topology: &NumaTopology, core: usize) -> Vec<usize> {
    let node = node_of_core(topology, core);
    cores_in_node(topology, node)
}

/// Cache-hierarchy distance between two cores.
///
/// | Value | Meaning |
/// |-------|---------|
/// | 0     | Same core (identity). |
/// | 1     | Different cores sharing the same L3 (same NUMA node). |
/// | 2     | Cores on different NUMA nodes (different L3 domains). |
pub fn cache_distance(topology: &NumaTopology, core_a: usize, core_b: usize) -> usize {
    if core_a == core_b {
        return 0;
    }
    let node_a = node_of_core(topology, core_a);
    let node_b = node_of_core(topology, core_b);
    if node_a == node_b {
        1
    } else {
        2
    }
}

// ─── Lookup helpers used by the scheduler ────────────────────────────────────

/// Return the `CoreInfo` for a given core ID, if it exists.
pub fn core_info(topology: &NumaTopology, core_id: usize) -> Option<&CoreInfo> {
    topology.cores.iter().find(|c| c.core_id == core_id)
}

/// Total number of cores across all NUMA nodes.
pub fn total_cores(topology: &NumaTopology) -> usize {
    topology.cores.len()
}

/// Return whether `core` is the first core in its NUMA node (used for
/// tie-breaking in steal decisions).
pub fn is_node_leader(topology: &NumaTopology, core: usize) -> bool {
    let node = node_of_core(topology, core);
    cores_in_node(topology, node)
        .first()
        .map(|&first| first == core)
        .unwrap_or(false)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn topo_2x4() -> NumaTopology {
        NumaTopology::from_config(2, 4)
    }

    #[test]
    fn test_topology_construction() {
        let t = topo_2x4();
        assert_eq!(t.n_nodes, 2);
        assert_eq!(t.n_cores_per_node, 4);
        assert_eq!(t.cores.len(), 8);
    }

    #[test]
    fn test_cores_in_node() {
        let t = topo_2x4();
        let node0 = cores_in_node(&t, 0);
        assert_eq!(node0, vec![0, 1, 2, 3]);
        let node1 = cores_in_node(&t, 1);
        assert_eq!(node1, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_node_of_core() {
        let t = topo_2x4();
        for c in 0..4 {
            assert_eq!(node_of_core(&t, c), 0, "core {} should be in node 0", c);
        }
        for c in 4..8 {
            assert_eq!(node_of_core(&t, c), 1, "core {} should be in node 1", c);
        }
    }

    #[test]
    fn test_distance_local() {
        assert_eq!(distance(0, 0), 10);
        assert_eq!(distance(1, 1), 10);
    }

    #[test]
    fn test_distance_remote() {
        assert_eq!(distance(0, 1), 20);
        assert_eq!(distance(1, 0), 20);
    }

    #[test]
    fn test_nearest_numa_sorted() {
        let t = topo_2x4();
        let order = nearest_numa_nodes(&t, 0);
        // Node 0 is nearest to itself (distance=10); node 1 is further (20).
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn test_cache_distance_same_core() {
        let t = topo_2x4();
        assert_eq!(cache_distance(&t, 2, 2), 0);
    }

    #[test]
    fn test_cache_distance_same_l3() {
        let t = topo_2x4();
        // Cores 0 and 3 are on node 0 — share L3.
        assert_eq!(cache_distance(&t, 0, 3), 1);
    }

    #[test]
    fn test_cache_distance_cross_numa() {
        let t = topo_2x4();
        // Core 0 (node 0) vs core 4 (node 1) → different L3.
        assert_eq!(cache_distance(&t, 0, 4), 2);
    }

    #[test]
    fn test_shared_l3_cores() {
        let t = topo_2x4();
        let shared = shared_l3_cores(&t, 0);
        // All four cores on node 0 share L3.
        assert_eq!(shared.len(), 4);
        assert!(shared.contains(&0));
        assert!(shared.contains(&3));
        assert!(!shared.contains(&4));
    }

    #[test]
    fn test_detect_topology_default() {
        let t = detect_topology();
        assert_eq!(t.n_nodes, 2);
        assert_eq!(t.n_cores_per_node, 4);
    }

    #[test]
    fn test_total_cores() {
        let t = NumaTopology::from_config(4, 8);
        assert_eq!(total_cores(&t), 32);
    }

    #[test]
    fn test_is_node_leader() {
        let t = topo_2x4();
        assert!(is_node_leader(&t, 0)); // first core of node 0
        assert!(!is_node_leader(&t, 1));
        assert!(is_node_leader(&t, 4)); // first core of node 1
        assert!(!is_node_leader(&t, 5));
    }
}
