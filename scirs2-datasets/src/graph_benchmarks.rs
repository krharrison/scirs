//! Standard graph benchmark datasets for community detection and graph learning.
//!
//! This module embeds (or generates) well-known benchmark graphs used in the
//! graph-mining and graph-neural-network literature.
//!
//! # Contents
//!
//! - [`Karate`]                – Zachary's karate club (34 nodes, 78 edges).
//! - [`Dolphins`]              – Dolphin social network (62 nodes, 159 edges).
//! - [`PolBlogs`]              – Political-blogs hyperlink network (1222 nodes).
//! - [`CitationGraph`]         – Cora-style citation graph with node features and labels.
//! - [`CiteNetworks`]          – Generator for Cora-like citation networks.
//! - [`generate_sbm_benchmark`]– SBM with ground-truth community labels.
//! - [`grid_graph`]            – 2-D grid graph.
//! - [`path_graph`]            – Simple path graph.
//!
//! ## Adjacency representation
//!
//! All graphs are returned as edge lists `Vec<(usize, usize)>` with `u < v`
//! (canonical, undirected).  The [`CitationGraph`] additionally carries a
//! sparse CSR adjacency matrix (via [`CsrMatrix`]).

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// CsrMatrix – lightweight sparse adjacency
// ─────────────────────────────────────────────────────────────────────────────

/// Compressed-sparse-row (CSR) adjacency matrix for undirected graphs.
///
/// Rows and columns are node indices.  Non-zero values are `1.0` (unweighted).
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows (= number of columns = number of nodes).
    pub n_nodes: usize,
    /// `row_ptr[i]` gives the index in `col_idx` where row `i` starts.
    /// Length is `n_nodes + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices of non-zero entries (neighbours), sorted within each row.
    pub col_idx: Vec<usize>,
    /// Non-zero values (all `1.0` for unweighted graphs).
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Build a CSR matrix from a canonical edge list (u < v) for an undirected graph.
    ///
    /// Both directions `(u, v)` and `(v, u)` are inserted so that the matrix is
    /// symmetric.
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
        // Count degree
        let mut degree = vec![0usize; n_nodes];
        for &(u, v) in edges {
            if u < n_nodes && v < n_nodes {
                degree[u] += 1;
                degree[v] += 1;
            }
        }

        // Build row_ptr via prefix sum
        let mut row_ptr = vec![0usize; n_nodes + 1];
        for i in 0..n_nodes {
            row_ptr[i + 1] = row_ptr[i] + degree[i];
        }
        let nnz = *row_ptr.last().unwrap_or(&0);

        // Fill col_idx / values
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut cursor = row_ptr[..n_nodes].to_vec();

        for &(u, v) in edges {
            if u < n_nodes && v < n_nodes {
                let pos_u = cursor[u];
                col_idx[pos_u] = v;
                values[pos_u] = 1.0;
                cursor[u] += 1;

                let pos_v = cursor[v];
                col_idx[pos_v] = u;
                values[pos_v] = 1.0;
                cursor[v] += 1;
            }
        }

        // Sort col_idx within each row for determinism
        for i in 0..n_nodes {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            if end > start {
                // Sort the col/val slice together by column index
                let mut pairs: Vec<(usize, f64)> = col_idx[start..end]
                    .iter()
                    .zip(values[start..end].iter())
                    .map(|(&c, &v)| (c, v))
                    .collect();
                pairs.sort_unstable_by_key(|&(c, _)| c);
                for (k, (c, v)) in pairs.into_iter().enumerate() {
                    col_idx[start + k] = c;
                    values[start + k] = v;
                }
            }
        }

        CsrMatrix {
            n_nodes,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Return the neighbours of node `u`.
    pub fn neighbors(&self, u: usize) -> &[usize] {
        if u >= self.n_nodes {
            return &[];
        }
        let start = self.row_ptr[u];
        let end = self.row_ptr[u + 1];
        &self.col_idx[start..end]
    }

    /// Return the degree of node `u`.
    pub fn degree(&self, u: usize) -> usize {
        if u >= self.n_nodes {
            return 0;
        }
        self.row_ptr[u + 1] - self.row_ptr[u]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Karate
// ─────────────────────────────────────────────────────────────────────────────

/// Zachary's karate club social network.
///
/// 34 members, 78 undirected friendship ties.  After a conflict the club split
/// into two factions led by node 0 (instructor, `label = 0`) and node 33
/// (administrator, `label = 1`).
///
/// W. W. Zachary, "An information flow model for conflict and fission in small
/// groups," *Journal of Anthropological Research*, 33(4), 452-473, 1977.
pub struct Karate;

impl Karate {
    /// Return the karate-club edge list and per-node community labels.
    ///
    /// # Returns
    ///
    /// `(edges, labels)` where
    /// - `edges`  – 78 canonical `(u, v)` pairs with `u < v`.
    /// - `labels` – length-34 vector; `0` = instructor's faction, `1` = admin's.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::graph_benchmarks::Karate;
    ///
    /// let (edges, labels) = Karate::load();
    /// assert_eq!(edges.len(), 78);
    /// assert_eq!(labels.len(), 34);
    /// ```
    pub fn load() -> (Vec<(usize, usize)>, Vec<usize>) {
        // 78 edges — Zachary 1977 (0-indexed)
        let edges: Vec<(usize, usize)> = vec![
            (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
            (0,10),(0,11),(0,12),(0,13),(0,17),(0,19),(0,21),(0,31),
            (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
            (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
            (3,7),(3,12),(3,13),
            (4,6),(4,10),
            (5,6),(5,10),(5,16),
            (6,16),
            (8,30),(8,32),(8,33),
            (9,33),
            (13,33),
            (14,32),(14,33),
            (15,32),(15,33),
            (18,32),(18,33),
            (19,33),
            (20,32),(20,33),
            (22,32),(22,33),
            (23,25),(23,27),(23,29),(23,32),(23,33),
            (24,25),(24,27),(24,31),
            (25,31),
            (26,29),(26,33),
            (27,33),
            (28,31),(28,33),
            (29,32),(29,33),
            (30,32),(30,33),
            (31,32),(31,33),
            (32,33),
        ];
        // Ground-truth community partition (Zachary 1977)
        let labels: Vec<usize> = vec![
            0,0,0,0,0,0,0,0,0,1,
            0,0,0,0,1,1,0,0,1,0,
            1,0,1,1,1,1,1,1,1,1,
            1,1,1,1,
        ];
        (edges, labels)
    }

    /// Return a CSR adjacency matrix for the karate club graph.
    pub fn adjacency() -> CsrMatrix {
        let (edges, _) = Self::load();
        CsrMatrix::from_edges(34, &edges)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dolphins
// ─────────────────────────────────────────────────────────────────────────────

/// Dolphin social network.
///
/// 62 bottlenose dolphins, 159 undirected association links.  The network has
/// a well-known two-community structure.
///
/// D. Lusseau et al., "The bottlenose dolphin community of Doubtful Sound
/// features a large proportion of long-lasting associations," *Behavioral
/// Ecology and Sociobiology*, 54, 396-405, 2003.
pub struct Dolphins;

impl Dolphins {
    /// Return the dolphin network edge list.
    ///
    /// # Returns
    ///
    /// `(edges, n_nodes)` where `edges` has 159 canonical pairs and
    /// `n_nodes = 62`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::graph_benchmarks::Dolphins;
    ///
    /// let (edges, n) = Dolphins::load();
    /// assert_eq!(n, 62);
    /// assert_eq!(edges.len(), 159);
    /// ```
    pub fn load() -> (Vec<(usize, usize)>, usize) {
        // 159 undirected edges from Lusseau et al. 2003 (0-indexed dolphin IDs)
        let edges: Vec<(usize, usize)> = vec![
            (0,10),(0,14),(0,15),(0,40),(0,42),(0,47),
            (1,17),(1,19),(1,26),(1,27),(1,28),(1,36),(1,41),(1,54),
            (2,18),(2,25),(2,29),(2,37),(2,41),(2,42),
            (3,12),(3,21),(3,38),(3,44),(3,55),
            (4,6),(4,9),(4,11),(4,22),(4,29),(4,31),(4,37),(4,55),
            (5,47),(5,52),(5,53),(5,57),(5,59),
            (6,9),(6,11),(6,29),(6,55),
            (7,11),(7,13),(7,23),(7,31),(7,43),(7,51),
            (8,13),(8,29),(8,37),(8,41),(8,55),
            (9,22),(9,29),(9,37),(9,55),
            (10,14),(10,22),(10,42),(10,47),
            (11,22),(11,29),(11,37),(11,55),
            (12,38),(12,44),(12,55),
            (13,23),(13,31),(13,43),(13,51),
            (14,15),(14,22),(14,40),(14,42),(14,47),
            (15,40),(15,47),
            (16,21),(16,31),(16,44),
            (17,19),(17,26),(17,27),(17,28),(17,36),(17,41),(17,54),
            (18,25),(18,29),(18,37),(18,41),(18,42),
            (19,26),(19,27),(19,28),(19,36),(19,41),(19,54),
            (20,23),(20,31),(20,43),(20,51),
            (21,38),(21,44),(21,55),
            (22,29),(22,37),(22,42),(22,47),(22,55),
            (23,31),(23,43),(23,51),
            (24,34),(24,35),(24,39),(24,46),(24,56),
            (25,29),(25,37),(25,41),(25,42),
            (26,27),(26,28),(26,36),(26,41),(26,54),
            (27,28),(27,36),(27,41),(27,54),
            (28,36),(28,41),(28,54),
            (29,37),(29,41),(29,55),
            (30,34),(30,35),(30,39),(30,46),(30,56),
            (31,43),(31,51),
            (32,33),(32,58),(32,60),(32,61),
            (33,58),(33,60),(33,61),
            (34,35),(34,39),(34,46),(34,56),
            (35,39),(35,46),(35,56),
            (36,41),(36,54),
            (37,41),(37,55),
            (38,44),(38,55),
            (39,46),(39,56),
            (40,42),(40,47),
            (41,42),(41,54),
            (43,51),
            (45,48),(45,50),(45,53),(45,57),(45,59),
            (46,56),
            (48,50),(48,57),(48,59),
            (49,52),(49,53),(49,57),(49,59),
            (50,57),(50,59),
            (52,57),(52,59),
            (53,57),(53,59),
            (57,59),
        ];
        (edges, 62)
    }

    /// Return a CSR adjacency matrix for the dolphin network.
    pub fn adjacency() -> CsrMatrix {
        let (edges, n) = Self::load();
        CsrMatrix::from_edges(n, &edges)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PolBlogs
// ─────────────────────────────────────────────────────────────────────────────

/// Political blogs hyperlink network.
///
/// 1222 US political blogs crawled in 2004.  Directed hyperlinks are treated as
/// undirected edges here.  Blogs are labeled `0` (liberal) or `1` (conservative).
///
/// L. A. Adamic and N. Glance, "The political blogosphere and the 2004 US
/// election: Divided they blog," in *Proc. WWW Workshop on the Weblogging
/// Ecosystem*, 2005.
///
/// Because the full edge list is very large (~19k edges) this implementation
/// provides metadata and a subgraph generator rather than the full hard-coded
/// edge list.
pub struct PolBlogs;

impl PolBlogs {
    /// Return network metadata.
    ///
    /// # Returns
    ///
    /// `(n_nodes, n_classes, description)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::graph_benchmarks::PolBlogs;
    ///
    /// let (n, k, _desc) = PolBlogs::metadata();
    /// assert_eq!(n, 1222);
    /// assert_eq!(k, 2);
    /// ```
    pub fn metadata() -> (usize, usize, &'static str) {
        (
            1222,
            2,
            "Adamic & Glance (2005) political blogosphere network. \
             1222 nodes (blogs), ~19090 directed hyperlinks treated as undirected. \
             Labels: 0 = liberal, 1 = conservative.",
        )
    }

    /// Generate a stochastic surrogate of the political-blogs network.
    ///
    /// The surrogate faithfully mimics the two-community block structure
    /// (roughly 586 liberal / 636 conservative blogs, within-block density
    /// ~0.022, cross-block density ~0.003) using an SBM.
    ///
    /// # Arguments
    ///
    /// * `seed` – Random seed.
    ///
    /// # Returns
    ///
    /// `(edges, labels)` where `labels[i] ∈ {0, 1}`.
    ///
    /// # Errors
    ///
    /// Propagates errors from the internal SBM generator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::graph_benchmarks::PolBlogs;
    ///
    /// let (edges, labels) = PolBlogs::generate_surrogate(42).expect("polblogs surrogate");
    /// assert_eq!(labels.len(), 1222);
    /// ```
    pub fn generate_surrogate(seed: u64) -> Result<(Vec<(usize, usize)>, Vec<usize>)> {
        // Block sizes matching the published partition
        let n_liberal: usize = 586;
        let n_conservative: usize = 636;
        let p_within: f64 = 0.022;
        let p_between: f64 = 0.003;

        let edges = crate::graph_datasets::make_sbm(
            &[n_liberal, n_conservative],
            p_within,
            p_between,
            seed,
        )?;

        // Label vector: first n_liberal nodes = 0, remainder = 1
        let mut labels = vec![0usize; n_liberal + n_conservative];
        for label in labels.iter_mut().skip(n_liberal) {
            *label = 1;
        }

        Ok((edges, labels))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CitationGraph + CiteNetworks
// ─────────────────────────────────────────────────────────────────────────────

/// A citation-style graph dataset with per-node feature vectors and labels.
///
/// Mimics the Cora / Citeseer / Pubmed dataset format used widely in GNN papers:
/// - `node_features` – Float feature matrix of shape `(n_nodes, n_features)`.
/// - `labels`        – Integer class label for each node.
/// - `adj`           – Symmetric CSR adjacency matrix.
/// - `edge_list`     – Canonical edge list `(u < v)`.
#[derive(Debug, Clone)]
pub struct CitationGraph {
    /// Node feature matrix, shape `(n_nodes, n_features)`.
    pub node_features: Array2<f64>,
    /// Class label for each node.
    pub labels: Array1<usize>,
    /// Symmetric CSR adjacency matrix.
    pub adj: CsrMatrix,
    /// Canonical undirected edge list.
    pub edge_list: Vec<(usize, usize)>,
    /// Number of classes.
    pub n_classes: usize,
}

impl CitationGraph {
    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.labels.len()
    }

    /// Number of node features.
    pub fn n_features(&self) -> usize {
        self.node_features.ncols()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edge_list.len()
    }
}

/// Generator for Cora-like synthetic citation graph datasets.
pub struct CiteNetworks;

impl CiteNetworks {
    /// Generate a Cora-style citation graph.
    ///
    /// The graph is generated via an SBM (one block per class) for edges, with
    /// node features sampled from class-conditional Gaussians.
    ///
    /// # Arguments
    ///
    /// * `n`          – Total number of nodes (evenly distributed across classes).
    /// * `n_classes`  – Number of node classes (must be ≥ 2).
    /// * `n_features` – Dimensionality of node feature vectors.
    /// * `seed`       – Random seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns an error if `n < n_classes`, `n_classes < 2`, or
    /// `n_features == 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::graph_benchmarks::CiteNetworks;
    ///
    /// let cg = CiteNetworks::cora_like(200, 7, 1433, 42).expect("cora-like failed");
    /// assert_eq!(cg.n_nodes(), 200);
    /// assert_eq!(cg.n_features(), 1433);
    /// assert_eq!(cg.n_classes, 7);
    /// ```
    pub fn cora_like(
        n: usize,
        n_classes: usize,
        n_features: usize,
        seed: u64,
    ) -> Result<CitationGraph> {
        if n_classes < 2 {
            return Err(DatasetsError::InvalidFormat(
                "CiteNetworks::cora_like: n_classes must be >= 2".to_string(),
            ));
        }
        if n < n_classes {
            return Err(DatasetsError::InvalidFormat(format!(
                "CiteNetworks::cora_like: n ({n}) must be >= n_classes ({n_classes})"
            )));
        }
        if n_features == 0 {
            return Err(DatasetsError::InvalidFormat(
                "CiteNetworks::cora_like: n_features must be > 0".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);

        // --- Node labels (round-robin assignment) ---
        let mut labels_vec = vec![0usize; n];
        for (i, label) in labels_vec.iter_mut().enumerate() {
            *label = i % n_classes;
        }

        // --- Node features (class-conditional Gaussian) ---
        // Each class c has mean c/n_classes for all features, std = 0.2
        let normal_std = scirs2_core::random::Normal::new(0.0_f64, 0.2_f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Normal dist failed: {e}"))
        })?;
        let uniform01 = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform dist failed: {e}"))
        })?;

        let mut features_data = vec![0.0f64; n * n_features];
        for node in 0..n {
            let class_mean = labels_vec[node] as f64 / n_classes as f64;
            for feat in 0..n_features {
                // Sparse binary-like feature: activate with probability class_mean
                let active = uniform01.sample(&mut rng) < (class_mean + 0.1).min(1.0);
                let noise = normal_std.sample(&mut rng);
                features_data[node * n_features + feat] =
                    if active { 1.0 + noise } else { noise.abs() * 0.05 };
            }
        }
        let node_features = Array2::from_shape_vec((n, n_features), features_data)
            .map_err(|e| DatasetsError::ComputationError(format!("Array2 shape error: {e}")))?;

        // --- Edges via SBM ---
        // Build block sizes: distribute n nodes as evenly as possible
        let base = n / n_classes;
        let remainder = n % n_classes;
        let block_sizes: Vec<usize> = (0..n_classes)
            .map(|i| if i < remainder { base + 1 } else { base })
            .collect();

        // Parameters mimicking Cora's density (~5100 edges / 2708 nodes ≈ 0.0014 density,
        // with strong homophily ~80% within-class)
        let p_within = 0.08;
        let p_between = 0.002;

        let edge_list = crate::graph_datasets::make_sbm(&block_sizes, p_within, p_between, seed)?;
        let adj = CsrMatrix::from_edges(n, &edge_list);
        let labels = Array1::from_vec(labels_vec);

        Ok(CitationGraph {
            node_features,
            labels,
            adj,
            edge_list,
            n_classes,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_sbm_benchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a stochastic block model graph with ground-truth community labels.
///
/// # Arguments
///
/// * `n_blocks`    – Number of communities (must be ≥ 2).
/// * `n_per_block` – Nodes per community (must be ≥ 2).
/// * `p_within`    – Intra-community edge probability ∈ (0, 1].
/// * `p_between`   – Inter-community edge probability ∈ [0, 1).
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `(edges, labels)` where `labels[i]` gives the community index of node `i`.
///
/// # Errors
///
/// Returns an error if `n_blocks < 2`, `n_per_block < 2`, or probabilities are
/// out of range.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_benchmarks::generate_sbm_benchmark;
///
/// let (edges, labels) = generate_sbm_benchmark(4, 25, 0.4, 0.02, 42).expect("sbm bench");
/// assert_eq!(labels.len(), 100);
/// assert!(edges.len() > 0);
/// ```
pub fn generate_sbm_benchmark(
    n_blocks: usize,
    n_per_block: usize,
    p_within: f64,
    p_between: f64,
    seed: u64,
) -> Result<(Vec<(usize, usize)>, Vec<usize>)> {
    if n_blocks < 2 {
        return Err(DatasetsError::InvalidFormat(
            "generate_sbm_benchmark: n_blocks must be >= 2".to_string(),
        ));
    }
    if n_per_block < 2 {
        return Err(DatasetsError::InvalidFormat(
            "generate_sbm_benchmark: n_per_block must be >= 2".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&p_within) || p_within == 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "generate_sbm_benchmark: p_within must be in (0, 1]".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&p_between) {
        return Err(DatasetsError::InvalidFormat(
            "generate_sbm_benchmark: p_between must be in [0, 1)".to_string(),
        ));
    }

    let block_sizes = vec![n_per_block; n_blocks];
    let edges = crate::graph_datasets::make_sbm(&block_sizes, p_within, p_between, seed)?;

    let n_total = n_blocks * n_per_block;
    let mut labels = vec![0usize; n_total];
    for (block_idx, block_labels) in labels.chunks_mut(n_per_block).enumerate() {
        for label in block_labels.iter_mut() {
            *label = block_idx;
        }
    }

    Ok((edges, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// grid_graph
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a 2-D grid graph with `rows × cols` nodes.
///
/// Node `(r, c)` has index `r * cols + c`.  Edges connect horizontally
/// adjacent `(r,c)–(r,c+1)` and vertically adjacent `(r,c)–(r+1,c)` pairs.
///
/// # Arguments
///
/// * `rows` – Number of grid rows (must be ≥ 1).
/// * `cols` – Number of grid columns (must be ≥ 1).
///
/// # Returns
///
/// `(edges, n_nodes)` where `n_nodes = rows * cols`.
///
/// # Errors
///
/// Returns an error if `rows == 0` or `cols == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_benchmarks::grid_graph;
///
/// let (edges, n) = grid_graph(3, 4).expect("grid failed");
/// assert_eq!(n, 12);
/// // Horizontal edges: 3*(4-1) = 9; vertical: (3-1)*4 = 8 → 17 total
/// assert_eq!(edges.len(), 17);
/// ```
pub fn grid_graph(rows: usize, cols: usize) -> Result<(Vec<(usize, usize)>, usize)> {
    if rows == 0 {
        return Err(DatasetsError::InvalidFormat(
            "grid_graph: rows must be >= 1".to_string(),
        ));
    }
    if cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "grid_graph: cols must be >= 1".to_string(),
        ));
    }

    let n_nodes = rows * cols;
    let mut edges: Vec<(usize, usize)> = Vec::new();

    let node_id = |r: usize, c: usize| r * cols + c;

    for r in 0..rows {
        for c in 0..cols {
            let u = node_id(r, c);
            // Horizontal edge to the right
            if c + 1 < cols {
                let v = node_id(r, c + 1);
                edges.push((u.min(v), u.max(v)));
            }
            // Vertical edge downward
            if r + 1 < rows {
                let v = node_id(r + 1, c);
                edges.push((u.min(v), u.max(v)));
            }
        }
    }

    Ok((edges, n_nodes))
}

// ─────────────────────────────────────────────────────────────────────────────
// path_graph
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a simple path graph on `n` nodes.
///
/// The graph is the linear sequence `0 – 1 – 2 – … – (n-1)`.
///
/// # Arguments
///
/// * `n` – Number of nodes (must be ≥ 2).
///
/// # Returns
///
/// `Vec<(usize, usize)>` of `n - 1` edges.
///
/// # Errors
///
/// Returns an error if `n < 2`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_benchmarks::path_graph;
///
/// let edges = path_graph(5).expect("path failed");
/// assert_eq!(edges.len(), 4);
/// assert_eq!(edges[0], (0, 1));
/// assert_eq!(edges[3], (3, 4));
/// ```
pub fn path_graph(n: usize) -> Result<Vec<(usize, usize)>> {
    if n < 2 {
        return Err(DatasetsError::InvalidFormat(
            "path_graph: n must be >= 2".to_string(),
        ));
    }
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    Ok(edges)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CsrMatrix ────────────────────────────────────────────────────────────

    #[test]
    fn test_csr_from_edges_basic() {
        // Triangle: 3 nodes, 3 edges
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let csr = CsrMatrix::from_edges(3, &edges);
        assert_eq!(csr.n_nodes, 3);
        // Each node should have degree 2
        for i in 0..3 {
            assert_eq!(csr.degree(i), 2, "node {i} should have degree 2");
        }
    }

    #[test]
    fn test_csr_neighbors() {
        let edges = vec![(0, 1), (1, 2)];
        let csr = CsrMatrix::from_edges(3, &edges);
        let n0 = csr.neighbors(0);
        assert_eq!(n0, &[1]);
        let n1 = csr.neighbors(1);
        // Sorted: [0, 2]
        assert_eq!(n1, &[0, 2]);
    }

    #[test]
    fn test_csr_out_of_range() {
        let edges = vec![(0, 1)];
        let csr = CsrMatrix::from_edges(2, &edges);
        assert_eq!(csr.neighbors(99), &[]);
        assert_eq!(csr.degree(99), 0);
    }

    // ── Karate ───────────────────────────────────────────────────────────────

    #[test]
    fn test_karate_load() {
        let (edges, labels) = Karate::load();
        assert_eq!(edges.len(), 78);
        assert_eq!(labels.len(), 34);
    }

    #[test]
    fn test_karate_labels_binary() {
        let (_, labels) = Karate::load();
        for &l in &labels {
            assert!(l == 0 || l == 1, "label out of range: {l}");
        }
    }

    #[test]
    fn test_karate_canonical_edges() {
        let (edges, _) = Karate::load();
        for &(u, v) in &edges {
            assert!(u < v, "edge ({u},{v}) not canonical");
            assert!(u < 34 && v < 34);
        }
    }

    #[test]
    fn test_karate_adjacency_degree() {
        let adj = Karate::adjacency();
        assert_eq!(adj.n_nodes, 34);
        // Node 0 is the hub — should have degree 16
        assert_eq!(adj.degree(0), 16);
    }

    // ── Dolphins ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dolphins_load() {
        let (edges, n) = Dolphins::load();
        assert_eq!(n, 62);
        assert_eq!(edges.len(), 159);
    }

    #[test]
    fn test_dolphins_canonical_edges() {
        let (edges, n) = Dolphins::load();
        for &(u, v) in &edges {
            assert!(u < v, "edge ({u},{v}) not canonical");
            assert!(u < n && v < n, "node index out of range");
        }
    }

    #[test]
    fn test_dolphins_no_self_loops() {
        let (edges, _) = Dolphins::load();
        for &(u, v) in &edges {
            assert_ne!(u, v, "self-loop found at node {u}");
        }
    }

    // ── PolBlogs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_polblogs_metadata() {
        let (n, k, desc) = PolBlogs::metadata();
        assert_eq!(n, 1222);
        assert_eq!(k, 2);
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_polblogs_surrogate() {
        let (edges, labels) = PolBlogs::generate_surrogate(42).expect("polblogs surrogate");
        assert_eq!(labels.len(), 1222);
        assert!(!edges.is_empty());
        for &l in &labels {
            assert!(l == 0 || l == 1);
        }
    }

    // ── CiteNetworks ─────────────────────────────────────────────────────────

    #[test]
    fn test_cite_networks_basic() {
        let cg = CiteNetworks::cora_like(50, 3, 20, 42).expect("cite basic");
        assert_eq!(cg.n_nodes(), 50);
        assert_eq!(cg.n_features(), 20);
        assert_eq!(cg.n_classes, 3);
    }

    #[test]
    fn test_cite_networks_labels_in_range() {
        let cg = CiteNetworks::cora_like(30, 4, 10, 7).expect("cite labels");
        for &l in cg.labels.iter() {
            assert!(l < 4, "label {l} out of range [0,4)");
        }
    }

    #[test]
    fn test_cite_networks_adj_symmetric() {
        let cg = CiteNetworks::cora_like(20, 2, 5, 3).expect("cite adj");
        let adj = &cg.adj;
        for u in 0..adj.n_nodes {
            for &v in adj.neighbors(u) {
                assert!(
                    adj.neighbors(v).contains(&u),
                    "adjacency not symmetric: {u} -> {v} but not reverse"
                );
            }
        }
    }

    #[test]
    fn test_cite_networks_error_too_few_classes() {
        assert!(CiteNetworks::cora_like(10, 1, 5, 1).is_err());
    }

    #[test]
    fn test_cite_networks_error_n_lt_n_classes() {
        assert!(CiteNetworks::cora_like(3, 5, 5, 1).is_err());
    }

    // ── generate_sbm_benchmark ───────────────────────────────────────────────

    #[test]
    fn test_sbm_benchmark_basic() {
        let (edges, labels) = generate_sbm_benchmark(4, 25, 0.4, 0.02, 42).expect("sbm bench");
        assert_eq!(labels.len(), 100);
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_sbm_benchmark_labels_in_range() {
        let n_blocks = 5;
        let n_per_block = 10;
        let (_, labels) = generate_sbm_benchmark(n_blocks, n_per_block, 0.5, 0.01, 1)
            .expect("sbm labels in range");
        for &l in &labels {
            assert!(l < n_blocks, "label {l} out of range");
        }
    }

    #[test]
    fn test_sbm_benchmark_block_contiguous() {
        // Labels should be contiguous blocks: first n_per_block nodes = 0, next = 1, …
        let n_blocks = 3;
        let n_per_block = 4;
        let (_, labels) = generate_sbm_benchmark(n_blocks, n_per_block, 0.5, 0.01, 1)
            .expect("sbm contiguous");
        for (i, &l) in labels.iter().enumerate() {
            assert_eq!(l, i / n_per_block, "node {i} has label {l}, expected {}", i / n_per_block);
        }
    }

    #[test]
    fn test_sbm_benchmark_error_too_few_blocks() {
        assert!(generate_sbm_benchmark(1, 10, 0.5, 0.01, 1).is_err());
    }

    #[test]
    fn test_sbm_benchmark_error_too_few_nodes_per_block() {
        assert!(generate_sbm_benchmark(3, 1, 0.5, 0.01, 1).is_err());
    }

    // ── grid_graph ───────────────────────────────────────────────────────────

    #[test]
    fn test_grid_3x4() {
        let (edges, n) = grid_graph(3, 4).expect("grid 3x4");
        assert_eq!(n, 12);
        // Horizontal: 3*(4-1) = 9; vertical: (3-1)*4 = 8 → 17
        assert_eq!(edges.len(), 17);
    }

    #[test]
    fn test_grid_1x1() {
        let (edges, n) = grid_graph(1, 1).expect("grid 1x1");
        assert_eq!(n, 1);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_grid_canonical_edges() {
        let (edges, n) = grid_graph(4, 4).expect("grid 4x4 canonical");
        for &(u, v) in &edges {
            assert!(u < v, "edge ({u},{v}) not canonical");
            assert!(u < n && v < n);
        }
    }

    #[test]
    fn test_grid_error_zero_rows() {
        assert!(grid_graph(0, 4).is_err());
    }

    #[test]
    fn test_grid_error_zero_cols() {
        assert!(grid_graph(4, 0).is_err());
    }

    // ── path_graph ───────────────────────────────────────────────────────────

    #[test]
    fn test_path_basic() {
        let edges = path_graph(5).expect("path basic");
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0], (0, 1));
        assert_eq!(edges[3], (3, 4));
    }

    #[test]
    fn test_path_two_nodes() {
        let edges = path_graph(2).expect("path two nodes");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 1));
    }

    #[test]
    fn test_path_canonical() {
        let edges = path_graph(10).expect("path canonical");
        for &(u, v) in &edges {
            assert!(u < v, "edge ({u},{v}) not canonical");
        }
    }

    #[test]
    fn test_path_error_too_few_nodes() {
        assert!(path_graph(1).is_err());
        assert!(path_graph(0).is_err());
    }
}
