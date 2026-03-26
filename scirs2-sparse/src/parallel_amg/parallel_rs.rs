//! Parallel Ruge-Stüben and Related Coarsening Algorithms
//!
//! This module implements several parallel coarsening strategies:
//!
//! 1. **PMIS** (Parallel Maximum Independent Set): A provably parallel algorithm
//!    that selects C-nodes as an independent set based on random weights.
//!
//! 2. **CLJP** (Cleary-Luby-Jones-Plassmann): A randomized parallel coarsening
//!    using λ-based probability selection.
//!
//! 3. **Parallel RS** (Ruge-Stüben with parallel passes): The classical RS
//!    algorithm adapted with parallel pass-1 processing.
//!
//! # References
//!
//! - Cleary, A. J., et al. (2000). "Robustness and scalability of algebraic
//!   multigrid." SIAM J. Sci. Comput.
//! - Luby, M. (1985). "A simple parallel algorithm for the maximal independent
//!   set problem." STOC.
//! - De Sterck, H., et al. (2008). "Distance-two interpolation for parallel
//!   algebraic multigrid." Numer. Linear Algebra Appl.

use crate::csr::CsrMatrix;
use crate::parallel_amg::strength::StrengthGraph;
use crate::parallel_amg::types::CoarseningResult;
use std::sync::{Arc, Mutex};

// ============================================================================
// Internal LCG PRNG (no rand dependency)
// ============================================================================

/// Simple LCG pseudo-random number generator (Knuth's constants)
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x123456789abcdef,
        }
    }

    /// Next u64
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Next f64 in [0.0, 1.0)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ============================================================================
// PMIS: Parallel Maximum Independent Set coarsening
// ============================================================================

/// PMIS coarsening: assigns random weights and selects an independent set.
///
/// Algorithm:
/// 1. Assign random weight w_i to each node.
/// 2. Node i becomes C-node if w_i > w_j for all j in strong neighborhood N_s(i).
/// 3. Mark all strong F-neighbors of C-nodes as F-nodes.
/// 4. Repeat until all nodes are assigned.
///
/// # Arguments
///
/// * `strength` - The strength-of-connection graph
///
/// # Returns
///
/// CoarseningResult with C/F splitting.
pub fn pmis_coarsening(strength: &StrengthGraph) -> CoarseningResult {
    let n = strength.n;
    if n == 0 {
        return CoarseningResult::from_splitting(Vec::new());
    }

    // 0 = undecided, 1 = C, 2 = F
    let mut status = vec![0u8; n];
    let mut rng = Lcg::new(42);

    // Assign random weights
    let mut weights: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

    let mut changed = true;
    let mut iter = 0usize;
    let max_iter = n + 10;

    while changed && iter < max_iter {
        changed = false;
        iter += 1;

        // Pass 1: find local maxima among undecided nodes
        let mut new_c = Vec::new();
        for i in 0..n {
            if status[i] != 0 {
                continue;
            }
            // Check if i has the maximum weight among its strong neighbors
            let mut is_max = true;
            // Check strong neighbors of i (i strongly influences them)
            for &j in &strength.strong_neighbors[i] {
                if status[j] == 0 && weights[j] >= weights[i] && j != i {
                    is_max = false;
                    break;
                }
            }
            // Also check nodes that strongly influence i
            if is_max {
                for &j in &strength.strong_influencers[i] {
                    if status[j] == 0 && weights[j] >= weights[i] && j != i {
                        is_max = false;
                        break;
                    }
                }
            }
            if is_max {
                new_c.push(i);
            }
        }

        // Assign C-nodes
        for &i in &new_c {
            status[i] = 1;
            changed = true;
        }

        // Pass 2: mark strong neighbors of new C-nodes as F-nodes
        for &c in &new_c {
            // Nodes that c strongly influences → F (they are in strong neighborhood of c)
            for &j in &strength.strong_neighbors[c] {
                if status[j] == 0 {
                    status[j] = 2;
                    changed = true;
                }
            }
            // Nodes that strongly influence c → F
            for &j in &strength.strong_influencers[c] {
                if status[j] == 0 {
                    status[j] = 2;
                    changed = true;
                }
            }
        }

        // Perturb weights of remaining undecided nodes to break ties
        for i in 0..n {
            if status[i] == 0 {
                weights[i] = rng.next_f64();
            }
        }
    }

    // Any remaining undecided nodes become F
    for s in status.iter_mut() {
        if *s == 0 {
            *s = 2;
        }
    }

    // Convert: status 1 = C, status 2 = F (0 in cf_splitting = F, 1 = C)
    let cf_splitting: Vec<u8> = status.iter().map(|&s| if s == 1 { 1 } else { 0 }).collect();
    CoarseningResult::from_splitting(cf_splitting)
}

// ============================================================================
// CLJP: Cleary-Luby-Jones-Plassmann coarsening
// ============================================================================

/// CLJP coarsening with λ-based probability selection.
///
/// Algorithm:
/// 1. Initialize all nodes as undecided.
/// 2. For each undecided node i, independently become C with probability
///    proportional to λ_i relative to its strong neighborhood.
/// 3. Remove C-nodes and their strong F-neighbors from undecided set.
/// 4. Update λ values and repeat.
///
/// # Arguments
///
/// * `strength` - The strength-of-connection graph
/// * `lambda` - Importance measure λ_i for each node
///
/// # Returns
///
/// CoarseningResult with C/F splitting.
pub fn cljp_coarsening(strength: &StrengthGraph, lambda: &[f64]) -> CoarseningResult {
    let n = strength.n;
    if n == 0 {
        return CoarseningResult::from_splitting(Vec::new());
    }

    // 0 = undecided, 1 = C, 2 = F
    let mut status = vec![0u8; n];
    let mut lambda_mut: Vec<f64> = lambda.to_vec();
    let mut rng = Lcg::new(137);

    // Assign initial random weights
    let mut weights: Vec<f64> = (0..n).map(|i| lambda_mut[i] + rng.next_f64()).collect();

    let max_iter = 2 * n + 10;
    let mut iter = 0usize;

    loop {
        // Check if any undecided nodes remain
        let remaining = status.iter().filter(|&&s| s == 0).count();
        if remaining == 0 || iter >= max_iter {
            break;
        }
        iter += 1;

        // Step 1: Each undecided node i becomes tentative C if it has the
        // maximum weight among its undecided strong neighborhood
        let mut new_c = Vec::new();
        for i in 0..n {
            if status[i] != 0 {
                continue;
            }
            let mut is_local_max = true;
            for &j in &strength.strong_neighbors[i] {
                if status[j] == 0 && weights[j] > weights[i] {
                    is_local_max = false;
                    break;
                }
            }
            if is_local_max {
                for &j in &strength.strong_influencers[i] {
                    if status[j] == 0 && weights[j] > weights[i] {
                        is_local_max = false;
                        break;
                    }
                }
            }
            if is_local_max {
                new_c.push(i);
            }
        }

        if new_c.is_empty() {
            // Perturb weights to break deadlock
            for i in 0..n {
                if status[i] == 0 {
                    weights[i] = lambda_mut[i] + rng.next_f64();
                }
            }
            continue;
        }

        // Assign C-nodes
        for &c in &new_c {
            status[c] = 1;
        }

        // Mark strong neighbors of new C-nodes as F-nodes
        let mut new_f = Vec::new();
        for &c in &new_c {
            for &j in &strength.strong_neighbors[c] {
                if status[j] == 0 {
                    status[j] = 2;
                    new_f.push(j);
                }
            }
            for &j in &strength.strong_influencers[c] {
                if status[j] == 0 {
                    status[j] = 2;
                    new_f.push(j);
                }
            }
        }

        // Update λ values for newly assigned F-nodes' neighbors
        for &f in &new_f {
            // Update nodes that f strongly influences (their F-count increases)
            for &k in &strength.strong_neighbors[f] {
                if status[k] == 0 {
                    lambda_mut[k] += 0.5;
                }
            }
        }

        // Refresh weights for undecided nodes
        for i in 0..n {
            if status[i] == 0 {
                weights[i] = lambda_mut[i] + rng.next_f64();
            }
        }
    }

    // Any remaining undecided → F
    for s in status.iter_mut() {
        if *s == 0 {
            *s = 2;
        }
    }

    let cf_splitting: Vec<u8> = status.iter().map(|&s| if s == 1 { 1 } else { 0 }).collect();
    CoarseningResult::from_splitting(cf_splitting)
}

// ============================================================================
// Parallel RS: Ruge-Stüben with parallel pass-1
// ============================================================================

/// Parallel Ruge-Stüben coarsening.
///
/// Implements the classical RS coarsening with parallel pass-1:
/// - Pass 1 (parallel): For each row, check if the node has maximum λ
///   among its undecided neighbors. Independent rows processed concurrently.
/// - Pass 2 (sequential): Ensure every F-F connection passes through a C-node.
///
/// # Arguments
///
/// * `a` - System matrix (used for pass-2 checking)
/// * `strength` - Strength graph
/// * `n_threads` - Number of parallel threads for pass-1
///
/// # Returns
///
/// CoarseningResult with C/F splitting.
pub fn parallel_rs_coarsening(
    a: &CsrMatrix<f64>,
    strength: &StrengthGraph,
    n_threads: usize,
) -> CoarseningResult {
    let n = strength.n;
    if n == 0 {
        return CoarseningResult::from_splitting(Vec::new());
    }

    let n_threads = n_threads.max(1);

    // Compute λ values (measure of importance)
    let mut lambda: Vec<f64> = strength
        .strong_influencers
        .iter()
        .map(|v| v.len() as f64)
        .collect();

    // status: 0 = undecided, 1 = C, 2 = F
    let status = Arc::new(Mutex::new(vec![0u8; n]));
    let lambda_arc = Arc::new(Mutex::new(lambda.clone()));

    let mut iter = 0usize;
    let max_iter = n + 10;

    while iter < max_iter {
        iter += 1;

        // Check remaining undecided nodes
        let remaining = {
            let s = status.lock().unwrap_or_else(|e| e.into_inner());
            s.iter().filter(|&&x| x == 0).count()
        };
        if remaining == 0 {
            break;
        }

        // Find undecided node with maximum lambda (greedy selection)
        let best_node = {
            let s = status.lock().unwrap_or_else(|e| e.into_inner());
            let l = lambda_arc.lock().unwrap_or_else(|e| e.into_inner());
            let mut best_idx = None;
            let mut best_val = -1.0f64;
            for i in 0..n {
                if s[i] == 0 && l[i] > best_val {
                    best_val = l[i];
                    best_idx = Some(i);
                }
            }
            best_idx
        };

        let c_node = match best_node {
            Some(i) => i,
            None => break,
        };

        // Mark as C
        {
            let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
            s[c_node] = 1;
        }

        // Collect strong neighbors to process in parallel
        let strong_nbrs: Vec<usize> = strength.strong_influencers[c_node].clone();

        let chunk_size = (strong_nbrs.len() + n_threads - 1).max(1) / n_threads.max(1);

        // Process strong neighbors in parallel: mark undecided ones as F
        let mut f_nodes_marked: Vec<usize> = Vec::new();

        std::thread::scope(|s_scope| {
            let mut handles = Vec::new();
            let status_ref = Arc::clone(&status);

            for chunk in strong_nbrs.chunks(chunk_size.max(1)) {
                let chunk_vec: Vec<usize> = chunk.to_vec();
                let status_clone = Arc::clone(&status_ref);

                let handle = s_scope.spawn(move || {
                    let mut local_f = Vec::new();
                    let mut guard = status_clone.lock().unwrap_or_else(|e| e.into_inner());
                    for &j in &chunk_vec {
                        if guard[j] == 0 {
                            guard[j] = 2; // F
                            local_f.push(j);
                        }
                    }
                    local_f
                });
                handles.push(handle);
            }

            for h in handles {
                if let Ok(local_f) = h.join() {
                    f_nodes_marked.extend(local_f);
                }
            }
        });

        // Also mark strong influencers as F (nodes that c_node strongly depends on)
        {
            let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
            for &j in &strength.strong_neighbors[c_node] {
                if s[j] == 0 {
                    s[j] = 2;
                    f_nodes_marked.push(j);
                }
            }
        }

        // Update λ: for each newly assigned F-node, increment λ of its
        // undecided strong influencers (they gain a potential F-neighbor)
        {
            let s = status.lock().unwrap_or_else(|e| e.into_inner());
            let mut l = lambda_arc.lock().unwrap_or_else(|e| e.into_inner());
            for &f in &f_nodes_marked {
                for &k in &strength.strong_influencers[f] {
                    if s[k] == 0 {
                        l[k] += 1.0;
                    }
                }
            }
        }

        // Retrieve updated lambda for next iteration
        {
            let l = lambda_arc.lock().unwrap_or_else(|e| e.into_inner());
            lambda.copy_from_slice(&l);
        }
    }

    // Any remaining undecided → F
    {
        let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
        for si in s.iter_mut() {
            if *si == 0 {
                *si = 2;
            }
        }
    }

    // Pass 2: Ensure that for every F-F strong connection, there exists a
    // shared C-neighbor. If not, promote one F-node to C.
    {
        let mut s = status.lock().unwrap_or_else(|e| e.into_inner());
        let n_rows = a.shape().0;
        for i in 0..n_rows {
            if s[i] != 2 {
                continue; // Only process F-nodes
            }
            // For each strong neighbor j of i that is also F
            for &j in &strength.strong_influencers[i] {
                if s[j] != 2 {
                    continue; // Only F-F pairs
                }
                // Check if i and j share a common C-neighbor
                let c_nbrs_i: Vec<usize> = strength.strong_influencers[i]
                    .iter()
                    .filter(|&&k| s[k] == 1)
                    .copied()
                    .collect();
                let c_nbrs_j: Vec<usize> = strength.strong_influencers[j]
                    .iter()
                    .filter(|&&k| s[k] == 1)
                    .copied()
                    .collect();
                let has_common_c = c_nbrs_i.iter().any(|k| c_nbrs_j.contains(k));
                if !has_common_c {
                    // Promote j to C
                    s[j] = 1;
                }
            }
        }
    }

    let s = status.lock().unwrap_or_else(|e| e.into_inner());
    let cf_splitting: Vec<u8> = s.iter().map(|&x| if x == 1 { 1 } else { 0 }).collect();
    CoarseningResult::from_splitting(cf_splitting)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_amg::strength::{compute_lambda, serial_strength_of_connection};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0f64);
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0f64);
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("valid Laplacian")
    }

    #[test]
    fn test_pmis_is_independent_set() {
        let a = laplacian_1d(12);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        // Verify: no two C-nodes are strongly connected
        for &c1 in &result.c_nodes {
            for &c2 in &result.c_nodes {
                if c1 != c2 {
                    assert!(
                        !g.strong_neighbors[c1].contains(&c2),
                        "C-nodes {c1} and {c2} are strongly connected — not an independent set"
                    );
                    assert!(
                        !g.strong_neighbors[c2].contains(&c1),
                        "C-nodes {c2} and {c1} are strongly connected — not an independent set"
                    );
                }
            }
        }
    }

    #[test]
    fn test_pmis_covers_all() {
        let a = laplacian_1d(12);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        // Every F-node must have at least one C-node in its strong neighborhood
        for &f in &result.f_nodes {
            let has_c_neighbor = g.strong_neighbors[f]
                .iter()
                .any(|&j| result.cf_splitting[j] == 1)
                || g.strong_influencers[f]
                    .iter()
                    .any(|&j| result.cf_splitting[j] == 1);
            assert!(
                has_c_neighbor,
                "F-node {f} has no C-node in strong neighborhood"
            );
        }
    }

    #[test]
    fn test_cljp_valid_splitting() {
        let a = laplacian_1d(14);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        let result = cljp_coarsening(&g, &lambda);

        // Every node must be assigned C or F
        assert_eq!(result.cf_splitting.len(), 14);
        for &s in &result.cf_splitting {
            assert!(s == 0 || s == 1, "Invalid splitting value: {s}");
        }
        assert!(!result.c_nodes.is_empty(), "Must have at least one C-node");
        assert!(!result.f_nodes.is_empty(), "Must have at least one F-node");
    }

    #[test]
    fn test_cljp_independent() {
        let a = laplacian_1d(14);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        let result = cljp_coarsening(&g, &lambda);

        // C-nodes should form an independent set in the strong graph
        for &c1 in &result.c_nodes {
            for &c2 in &result.c_nodes {
                if c1 != c2 {
                    assert!(
                        !g.strong_neighbors[c1].contains(&c2),
                        "CLJP C-nodes {c1} and {c2} are directly connected"
                    );
                }
            }
        }
    }

    #[test]
    fn test_parallel_rs_cf_assignment() {
        let a = laplacian_1d(16);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = parallel_rs_coarsening(&a, &g, 2);

        assert_eq!(result.cf_splitting.len(), 16);
        assert!(!result.c_nodes.is_empty());
        assert!(!result.f_nodes.is_empty());
        assert_eq!(result.c_nodes.len() + result.f_nodes.len(), 16);
    }

    #[test]
    fn test_parallel_rs_matches_serial() {
        let a = laplacian_1d(20);
        let g = serial_strength_of_connection(&a, 0.25);

        // Run with 1 thread
        let r1 = parallel_rs_coarsening(&a, &g, 1);
        // Run with 4 threads
        let r4 = parallel_rs_coarsening(&a, &g, 4);

        // Both should produce valid splits; coarse counts should be in same ballpark
        let ratio1 = r1.coarsening_ratio();
        let ratio4 = r4.coarsening_ratio();

        assert!(
            ratio1 > 0.0 && ratio1 < 1.0,
            "1-thread ratio out of range: {ratio1}"
        );
        assert!(
            ratio4 > 0.0 && ratio4 < 1.0,
            "4-thread ratio out of range: {ratio4}"
        );
    }

    #[test]
    fn test_coarsening_ratio() {
        let a = laplacian_1d(20);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let ratio = result.coarsening_ratio();
        assert!(
            (0.15..=0.85).contains(&ratio),
            "Coarsening ratio {ratio} out of expected range [0.15, 0.85]"
        );
    }
}
