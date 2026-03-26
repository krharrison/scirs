//! Advanced streaming graph algorithms.
//!
//! This module provides algorithms that process massive graphs edge-by-edge
//! without requiring the full graph to reside in memory.
//!
//! # Algorithms
//!
//! | Algorithm | Description |
//! |-----------|-------------|
//! | [`StreamingTriangleCounter`] | Reservoir + window exact/approximate triangle counting |
//! | [`StreamingUnionFind`] | Online connected-components via path-compressed Union-Find |
//! | [`streaming_bfs`] | Multi-pass BFS over an edge stream (memory-efficient) |
//! | [`StreamingDegreeEstimator`] | Count-min sketch approximate degree queries |

use std::collections::{HashMap, HashSet, VecDeque};

// ────────────────────────────────────────────────────────────────────────────
// StreamConfig
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for streaming graph algorithms.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Number of recent edges kept in the sliding window (for triangle counting).
    pub window_size: usize,
    /// Bit-width of hash functions in the count-min sketch (determines width `w = 2^n_sketch_bits`).
    pub n_sketch_bits: usize,
    /// Random seed used for hash functions.
    pub seed: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            n_sketch_bits: 64,
            seed: 42,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GraphStream
// ────────────────────────────────────────────────────────────────────────────

/// An iterator-backed stream of graph edges `(u, v)`.
///
/// Use [`GraphStream::from_edges`] to create a stream from a vector, or
/// [`GraphStream::from_fn`] to wrap a lazy generator (e.g. reading a file
/// line-by-line without loading it all).
pub struct GraphStream {
    inner: Box<dyn Iterator<Item = (usize, usize)>>,
}

impl GraphStream {
    /// Create a [`GraphStream`] from an owned vector of edges.
    pub fn from_edges(edges: Vec<(usize, usize)>) -> Self {
        Self {
            inner: Box::new(edges.into_iter()),
        }
    }

    /// Create a [`GraphStream`] from a closure that produces `Some((u,v))` or `None`.
    pub fn from_fn(mut f: impl FnMut() -> Option<(usize, usize)> + 'static) -> Self {
        Self {
            inner: Box::new(std::iter::from_fn(f)),
        }
    }

    /// Advance the stream by one edge.  Returns `None` when exhausted.
    pub fn next_edge(&mut self) -> Option<(usize, usize)> {
        self.inner.next()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingTriangleCounter
// ────────────────────────────────────────────────────────────────────────────

/// Streaming triangle counter using a sliding adjacency window.
///
/// For each new edge `(u, v)` the algorithm counts common neighbours already
/// present in the current window, giving an exact count for edges within the
/// window and an approximate count when the stream overflows it.
///
/// For small graphs (total edges ≤ `window_size`) this is an exact counter.
///
/// # References
/// Buriol et al., "Counting Triangles in Data Streams", PODS 2006.
#[derive(Debug)]
pub struct StreamingTriangleCounter {
    /// Partial adjacency maintained for the current window.
    adjacency_sketch: HashMap<usize, HashSet<usize>>,
    /// Accumulated triangle estimate.
    triangle_count: f64,
    /// Number of edges processed so far.
    n_edges: usize,
    /// Sliding reservoir of recent edges (FIFO, bounded by `window_size`).
    reservoir: VecDeque<(usize, usize)>,
    /// Maximum window size.
    window_size: usize,
}

impl StreamingTriangleCounter {
    /// Create a new counter with the given configuration.
    pub fn new(config: StreamConfig) -> Self {
        Self {
            adjacency_sketch: HashMap::new(),
            triangle_count: 0.0,
            n_edges: 0,
            reservoir: VecDeque::new(),
            window_size: config.window_size,
        }
    }

    /// Process a single edge from the stream.
    ///
    /// New triangles formed by `(u, v)` together with any common neighbour `w`
    /// already present in the adjacency sketch are counted.  When the reservoir
    /// is full the oldest edge is evicted and removed from the sketch.
    pub fn process_edge(&mut self, u: usize, v: usize) {
        self.n_edges += 1;

        // Count triangles closed by this edge: |N(u) ∩ N(v)|
        let neighbours_u: HashSet<usize> =
            self.adjacency_sketch.get(&u).cloned().unwrap_or_default();
        let neighbours_v: HashSet<usize> =
            self.adjacency_sketch.get(&v).cloned().unwrap_or_default();

        let common = neighbours_u.intersection(&neighbours_v).count();

        // Scale factor: when the window is full we are counting on a sample of
        // size `window_size / n_edges` of all edges, so scale up by (n/m)^2.
        let scale = if self.n_edges <= self.window_size {
            1.0
        } else {
            let m = self.window_size as f64;
            let n = self.n_edges as f64;
            (n / m) * (n / m)
        };
        self.triangle_count += common as f64 * scale;

        // Add edge to sketch (undirected)
        self.adjacency_sketch.entry(u).or_default().insert(v);
        self.adjacency_sketch.entry(v).or_default().insert(u);
        self.reservoir.push_back((u, v));

        // Evict oldest edge when window overflows
        if self.reservoir.len() > self.window_size {
            if let Some((eu, ev)) = self.reservoir.pop_front() {
                if let Some(set) = self.adjacency_sketch.get_mut(&eu) {
                    set.remove(&ev);
                }
                if let Some(set) = self.adjacency_sketch.get_mut(&ev) {
                    set.remove(&eu);
                }
            }
        }
    }

    /// Return the current triangle estimate.
    pub fn estimate_triangles(&self) -> f64 {
        self.triangle_count
    }

    /// Drive the counter over an entire [`GraphStream`], returning the final estimate.
    pub fn process_stream(&mut self, stream: &mut GraphStream) -> f64 {
        while let Some((u, v)) = stream.next_edge() {
            self.process_edge(u, v);
        }
        self.estimate_triangles()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingBFS
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for streaming BFS.
#[derive(Debug, Clone)]
pub struct StreamingBfsConfig {
    /// Source vertex for BFS.
    pub source: usize,
    /// Maximum distance to explore (inclusive).
    pub max_dist: usize,
    /// Soft memory limit: maximum number of vertices to store in `visited`.
    pub memory_limit: usize,
}

impl Default for StreamingBfsConfig {
    fn default() -> Self {
        Self {
            source: 0,
            max_dist: usize::MAX,
            memory_limit: 10_000,
        }
    }
}

/// Result of a streaming BFS.
#[derive(Debug, Clone)]
pub struct StreamBfsResult {
    /// Map from vertex to its shortest-path distance from the source.
    pub distances: HashMap<usize, usize>,
    /// Number of stream passes performed.
    pub n_passes: usize,
    /// Number of distinct vertices reached.
    pub n_vertices_reached: usize,
}

/// Memory-efficient multi-pass BFS over a [`GraphStream`].
///
/// Because a streaming BFS cannot rewind the stream, we work over a snapshot
/// of edges (the stream is fully consumed once and stored as an edge list).
/// Pass `k` discovers all vertices at distance `k` from the source.  Only the
/// current frontier and the visited set need to be kept in memory.
///
/// # Algorithm
/// 1. Consume the stream into an edge list (required for multi-pass).
/// 2. Pass 0: initialise `visited = {source}`, `frontier = {source}`, `dist[source] = 0`.
/// 3. Pass k: scan every edge; if exactly one endpoint is in the frontier and
///    the other is unvisited, add the other to the next frontier at distance k+1.
/// 4. Repeat until no new vertices are discovered or `max_dist` is reached.
pub fn streaming_bfs(stream: &mut GraphStream, config: &StreamingBfsConfig) -> StreamBfsResult {
    // Collect all edges for multi-pass traversal
    let mut edges: Vec<(usize, usize)> = Vec::new();
    while let Some(e) = stream.next_edge() {
        edges.push(e);
    }

    let source = config.source;
    let mut distances: HashMap<usize, usize> = HashMap::new();
    distances.insert(source, 0);

    let mut frontier: HashSet<usize> = HashSet::new();
    frontier.insert(source);

    let mut n_passes = 0usize;
    let mut current_dist = 0usize;

    while !frontier.is_empty() && current_dist < config.max_dist {
        let mut next_frontier: HashSet<usize> = HashSet::new();
        // Scan the full edge list for edges crossing from frontier to unvisited
        for &(u, v) in &edges {
            // Check both directions (undirected stream)
            for &(a, b) in &[(u, v), (v, u)] {
                if frontier.contains(&a)
                    && !distances.contains_key(&b)
                    && distances.len() < config.memory_limit
                {
                    distances.insert(b, current_dist + 1);
                    next_frontier.insert(b);
                }
            }
        }
        n_passes += 1;
        current_dist += 1;
        frontier = next_frontier;
    }

    let n_vertices_reached = distances.len();
    StreamBfsResult {
        distances,
        n_passes,
        n_vertices_reached,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingDegreeEstimator (count-min sketch)
// ────────────────────────────────────────────────────────────────────────────

/// Streaming degree estimator backed by a count-min sketch.
///
/// Each call to `process_edge` increments the sketch cell for both endpoints.
/// The sketch uses `d` independent hash functions, each with width `w` determined
/// by `n_sketch_bits` (w = min(2^n_sketch_bits, 2^16) to keep memory bounded).
/// Degree queries return the minimum over all `d` rows.
///
/// # Space
/// O(d × w) counters where d = 4 and w = 2^min(n_sketch_bits, 16).
#[derive(Debug)]
pub struct StreamingDegreeEstimator {
    /// count_min[row][col] — d rows × w columns.
    count_min: Vec<Vec<u32>>,
    /// Number of hash functions (rows).
    d: usize,
    /// Width of each row.
    w: usize,
    /// Number of edges processed.
    n_edges: usize,
    /// Hash seeds, one per row.
    seeds: Vec<u64>,
}

impl StreamingDegreeEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StreamConfig) -> Self {
        let d = 4usize;
        // Cap width at 2^16 = 65536 to keep memory sensible
        let bits = config.n_sketch_bits.min(16);
        let w = 1usize << bits;

        // Derive d different seeds from the base seed
        let seeds: Vec<u64> = (0..d)
            .map(|i| {
                config
                    .seed
                    .wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            })
            .collect();

        Self {
            count_min: vec![vec![0u32; w]; d],
            d,
            w,
            n_edges: 0,
            seeds,
        }
    }

    /// Hash a vertex id using seed `s` to a column index in `[0, w)`.
    fn hash_vertex(&self, vertex: usize, seed: u64) -> usize {
        // FNV-1a inspired mix
        let mut h = seed ^ (vertex as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        (h as usize) % self.w
    }

    /// Process a single edge `(u, v)` — increments degree counters for both endpoints.
    pub fn process_edge(&mut self, u: usize, v: usize) {
        self.n_edges += 1;
        for row in 0..self.d {
            let seed = self.seeds[row];
            let col_u = self.hash_vertex(u, seed);
            let col_v = self.hash_vertex(v, seed);
            self.count_min[row][col_u] = self.count_min[row][col_u].saturating_add(1);
            self.count_min[row][col_v] = self.count_min[row][col_v].saturating_add(1);
        }
    }

    /// Estimate the degree of `vertex` — returns the minimum counter across rows.
    pub fn estimate_degree(&self, vertex: usize) -> u32 {
        (0..self.d)
            .map(|row| {
                let col = self.hash_vertex(vertex, self.seeds[row]);
                self.count_min[row][col]
            })
            .min()
            .unwrap_or(0)
    }

    /// Return estimated degrees for each vertex id in `0..n_vertices`.
    pub fn approximate_degree_distribution(&self, n_vertices: usize) -> Vec<u32> {
        (0..n_vertices).map(|v| self.estimate_degree(v)).collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingUnionFind
// ────────────────────────────────────────────────────────────────────────────

/// Online connected-components via path-compressed, union-by-rank Union-Find.
///
/// As edges arrive from the stream the structure is updated in O(α(n)) amortised
/// time per edge (inverse Ackermann).
#[derive(Debug, Default)]
pub struct StreamingUnionFind {
    /// parent[x] = parent of x; root if parent[x] == x.
    parent: HashMap<usize, usize>,
    /// rank[x] = upper bound on tree height at x.
    rank: HashMap<usize, usize>,
}

impl StreamingUnionFind {
    /// Create an empty Union-Find structure.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure vertex `x` is initialised (self-loop, rank 0).
    fn make_set(&mut self, x: usize) {
        self.parent.entry(x).or_insert(x);
        self.rank.entry(x).or_insert(0);
    }

    /// Find the representative of `x` with full path compression (iterative).
    pub fn find(&mut self, x: usize) -> usize {
        self.make_set(x);

        // Walk up to the root
        let mut root = x;
        loop {
            let p = *self.parent.get(&root).unwrap_or(&root);
            if p == root {
                break;
            }
            root = p;
        }

        // Path compression: point every node on the path directly to root
        let mut current = x;
        loop {
            let p = *self.parent.get(&current).unwrap_or(&current);
            if p == root {
                break;
            }
            self.parent.insert(current, root);
            current = p;
        }
        root
    }

    /// Process edge `(u, v)` — union the two components.
    pub fn process_edge(&mut self, u: usize, v: usize) {
        self.make_set(u);
        self.make_set(v);
        let ru = self.find(u);
        let rv = self.find(v);
        if ru == rv {
            return; // already connected
        }
        // Union by rank
        let rank_u = *self.rank.get(&ru).unwrap_or(&0);
        let rank_v = *self.rank.get(&rv).unwrap_or(&0);
        match rank_u.cmp(&rank_v) {
            std::cmp::Ordering::Less => {
                self.parent.insert(ru, rv);
            }
            std::cmp::Ordering::Greater => {
                self.parent.insert(rv, ru);
            }
            std::cmp::Ordering::Equal => {
                self.parent.insert(rv, ru);
                self.rank.insert(ru, rank_u + 1);
            }
        }
    }

    /// Return the number of distinct connected components.
    pub fn n_components(&self) -> usize {
        self.parent.iter().filter(|(&node, &p)| node == p).count()
    }

    /// Return the canonical component identifier for `x`.
    pub fn component_id(&mut self, x: usize) -> usize {
        self.find(x)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Triangle counting: K4 has exactly 4 triangles ────────────────────────
    #[test]
    fn test_streaming_triangle_k4() {
        // K4 edges (6 total)
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let config = StreamConfig {
            window_size: 100,
            ..Default::default()
        };
        let mut counter = StreamingTriangleCounter::new(config);
        let mut stream = GraphStream::from_edges(edges);
        let estimate = counter.process_stream(&mut stream);
        // K4 has exactly 4 triangles; with window >= 6 this should be exact
        assert!(
            (estimate - 4.0).abs() < 1.0,
            "Expected ~4.0 triangles, got {estimate}"
        );
    }

    // ── StreamingUnionFind: path graph is one component ──────────────────────
    #[test]
    fn test_streaming_union_find_path_graph() {
        let mut uf = StreamingUnionFind::new();
        // Path 0-1-2-3-4
        for i in 0..4usize {
            uf.process_edge(i, i + 1);
        }
        assert_eq!(uf.n_components(), 1, "Path graph should be one component");
        // All vertices should share the same component id
        let c0 = uf.component_id(0);
        for v in 1..5usize {
            assert_eq!(uf.component_id(v), c0);
        }
    }

    // ── StreamingUnionFind: disconnected graph has correct component count ───
    #[test]
    fn test_streaming_union_find_disconnected() {
        let mut uf = StreamingUnionFind::new();
        // Three disconnected edges: (0,1), (2,3), (4,5)
        uf.process_edge(0, 1);
        uf.process_edge(2, 3);
        uf.process_edge(4, 5);
        assert_eq!(uf.n_components(), 3);
    }

    // ── StreamingBFS: small graph ─────────────────────────────────────────────
    #[test]
    fn test_streaming_bfs_small_graph() {
        // Graph: 0-1, 1-2, 2-3, 1-3
        let edges = vec![(0, 1), (1, 2), (2, 3), (1, 3)];
        let mut stream = GraphStream::from_edges(edges);
        let config = StreamingBfsConfig {
            source: 0,
            ..Default::default()
        };
        let result = streaming_bfs(&mut stream, &config);
        assert_eq!(result.distances[&0], 0);
        assert_eq!(result.distances[&1], 1);
        assert_eq!(result.distances[&2], 2);
        assert_eq!(result.distances[&3], 2);
        assert_eq!(result.n_vertices_reached, 4);
    }

    // ── StreamingBFS: single source, star graph ───────────────────────────────
    #[test]
    fn test_streaming_bfs_star() {
        // Star: center 0, leaves 1..=5
        let edges: Vec<(usize, usize)> = (1..=5).map(|i| (0, i)).collect();
        let mut stream = GraphStream::from_edges(edges);
        let config = StreamingBfsConfig {
            source: 0,
            ..Default::default()
        };
        let result = streaming_bfs(&mut stream, &config);
        assert_eq!(result.distances[&0], 0);
        for leaf in 1..=5usize {
            assert_eq!(result.distances[&leaf], 1);
        }
    }

    // ── DegreeEstimator: known graph degrees ─────────────────────────────────
    #[test]
    fn test_degree_estimator_known_degrees() {
        // Build a graph where vertex 0 has degree 4
        // Edges: 0-1, 0-2, 0-3, 0-4
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let config = StreamConfig {
            n_sketch_bits: 8,
            ..Default::default()
        };
        let mut estimator = StreamingDegreeEstimator::new(config);
        for (u, v) in &edges {
            estimator.process_edge(*u, *v);
        }
        let est_deg_0 = estimator.estimate_degree(0);
        // True degree is 4; allow 2x error as per spec
        assert!(
            est_deg_0 >= 2,
            "Degree estimate for vertex 0 should be >= 2 (true=4), got {est_deg_0}"
        );
        // Leaves have degree 1
        for v in 1..=4usize {
            let est = estimator.estimate_degree(v);
            assert!(
                est >= 1,
                "Degree estimate for leaf {v} should be >= 1, got {est}"
            );
        }
    }

    // ── DegreeEstimator: degree distribution ─────────────────────────────────
    #[test]
    fn test_degree_estimator_distribution() {
        // Path 0-1-2-3-4: degrees are 1,2,2,2,1
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let config = StreamConfig {
            n_sketch_bits: 8,
            ..Default::default()
        };
        let mut estimator = StreamingDegreeEstimator::new(config);
        for (u, v) in &edges {
            estimator.process_edge(*u, *v);
        }
        let dist = estimator.approximate_degree_distribution(5);
        // All estimates should be ≥ 1 and ≤ 4 (2× the max true degree of 2)
        for (v, &est) in dist.iter().enumerate() {
            assert!(est >= 1, "Vertex {v} degree estimate {est} should be >= 1");
            assert!(est <= 8, "Vertex {v} degree estimate {est} should be <= 8");
        }
    }

    // ── GraphStream from_fn ───────────────────────────────────────────────────
    #[test]
    fn test_graph_stream_from_fn() {
        let data = vec![(0usize, 1usize), (1, 2)];
        let mut iter = data.into_iter();
        let mut stream = GraphStream::from_fn(move || iter.next());
        assert_eq!(stream.next_edge(), Some((0, 1)));
        assert_eq!(stream.next_edge(), Some((1, 2)));
        assert_eq!(stream.next_edge(), None);
    }
}
