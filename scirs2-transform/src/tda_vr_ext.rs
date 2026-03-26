//! Extended Topological Data Analysis utilities
//!
//! This module provides additional TDA constructs and free functions that
//! complement the core `tda` module:
//!
//! - [`VietorisRipsComplex`]: explicit simplex-list representation with Euler
//!   characteristic and simplex counting.
//! - [`compute_persistence`]: distance-matrix-based persistent homology.
//! - [`persistence_landscape_fn`]: evaluate persistence landscapes over a grid.
//! - [`persistence_image_fn`]: Gaussian-kernel persistence image from a diagram.
//! - `wasserstein_distance_p`: p-Wasserstein distance between diagrams.
//! - `bottleneck_distance_fn`: bottleneck distance as a free function.
//!
//! ## References
//!
//! - Edelsbrunner & Harer (2010). Computational Topology: An Introduction.
//! - Adams et al. (2017). Persistence Images: A Stable Vector Representation.
//! - Munch (2017). A User's Guide to Topological Data Analysis.

use crate::error::{Result, TransformError};
use crate::tda_vr::{PersistenceDiagram, VietorisRips};
use scirs2_core::ndarray::Array2;

// ─── VietorisRipsComplex ─────────────────────────────────────────────────────

/// A Vietoris-Rips simplicial complex at a fixed scale `epsilon`.
///
/// Unlike [`VietorisRips`] (which computes the full persistent homology
/// across all scales), this struct stores the explicit list of simplices
/// formed when all pairwise distances ≤ `epsilon`.
///
/// ## Example
///
/// ```rust
/// use scirs2_transform::tda_ext::VietorisRipsComplex;
///
/// let pts = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![1.0, 1.0],
///     vec![0.0, 1.0],
/// ];
/// let vrc = VietorisRipsComplex::new(&pts, 1.5).expect("should succeed");
/// assert!(vrc.n_simplices(0) == 4); // four 0-simplices (vertices)
/// assert!(vrc.euler_characteristic() != 0); // non-trivial topology
/// ```
#[derive(Debug, Clone)]
pub struct VietorisRipsComplex {
    /// Input point cloud
    pub points: Vec<Vec<f64>>,
    /// Scale parameter (all edges with length ≤ epsilon are included)
    pub epsilon: f64,
    /// All simplices (sorted by dimension then by vertex tuple)
    pub simplices: Vec<Vec<usize>>,
}

impl VietorisRipsComplex {
    /// Construct the Vietoris-Rips complex for the given point cloud at scale `epsilon`.
    ///
    /// Only simplices up to dimension 2 (triangles) are computed for tractability.
    ///
    /// # Arguments
    /// * `points`  — slice of point vectors (all of equal length)
    /// * `epsilon` — maximum edge length to include
    pub fn new(points: &[Vec<f64>], epsilon: f64) -> Result<Self> {
        if points.is_empty() {
            return Ok(Self {
                points: Vec::new(),
                epsilon,
                simplices: Vec::new(),
            });
        }
        if epsilon < 0.0 {
            return Err(TransformError::InvalidInput(
                "epsilon must be non-negative".to_string(),
            ));
        }
        let n = points.len();
        let dim = points[0].len();

        // Pairwise distances
        let dist = pairwise_distances(points, dim);

        let mut simplices: Vec<Vec<usize>> = Vec::new();

        // 0-simplices: all vertices
        for i in 0..n {
            simplices.push(vec![i]);
        }

        // 1-simplices: all edges with dist <= epsilon
        for i in 0..n {
            for j in (i + 1)..n {
                if dist[i][j] <= epsilon {
                    simplices.push(vec![i, j]);
                }
            }
        }

        // 2-simplices: triangles (all edges present)
        for i in 0..n {
            for j in (i + 1)..n {
                if dist[i][j] > epsilon {
                    continue;
                }
                for k in (j + 1)..n {
                    if dist[i][k] <= epsilon && dist[j][k] <= epsilon {
                        simplices.push(vec![i, j, k]);
                    }
                }
            }
        }

        // Sort: by dimension first, then lexicographically within each dimension
        simplices.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        Ok(Self {
            points: points.to_vec(),
            epsilon,
            simplices,
        })
    }

    /// Count the number of simplices of a given dimension.
    ///
    /// Dimension 0 = vertices, 1 = edges, 2 = triangles.
    pub fn n_simplices(&self, dim: usize) -> usize {
        self.simplices.iter().filter(|s| s.len() == dim + 1).count()
    }

    /// Compute the Euler characteristic χ = Σ_k (-1)^k * C_k,
    /// where C_k is the number of k-simplices.
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0_i64;
        for simplex in &self.simplices {
            let k = simplex.len() as i64 - 1;
            if k % 2 == 0 {
                chi += 1;
            } else {
                chi -= 1;
            }
        }
        chi
    }

    /// Check whether two vertices are connected by an edge in the complex.
    pub fn are_connected(&self, u: usize, v: usize) -> bool {
        let edge = if u < v { vec![u, v] } else { vec![v, u] };
        self.simplices.contains(&edge)
    }

    /// List all edges (1-simplices) as pairs (u, v).
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.simplices
            .iter()
            .filter(|s| s.len() == 2)
            .map(|s| (s[0], s[1]))
            .collect()
    }
}

// ─── compute_persistence (distance matrix API) ────────────────────────────────

/// Compute persistent homology from a precomputed distance matrix.
///
/// Returns one [`PersistenceDiagram`] per homological dimension (H0, H1, …,
/// up to `max_dim`).  The filtration is the Vietoris-Rips filtration: a
/// simplex enters at the maximum pairwise distance among its vertices.
///
/// This is equivalent to constructing a nested family of Vietoris-Rips
/// complexes parameterised by `epsilon ∈ [0, max_epsilon]`.
///
/// # Arguments
/// * `distance_matrix` — symmetric n×n matrix of pairwise distances
/// * `max_dim`         — maximum homological dimension to compute (typically 1 or 2)
/// * `max_epsilon`     — upper bound on the filtration parameter
///
/// # Example
///
/// ```rust
/// use scirs2_transform::tda_ext::compute_persistence;
///
/// let dist = vec![
///     vec![0.0, 1.0, 1.4, 1.0],
///     vec![1.0, 0.0, 1.0, 1.4],
///     vec![1.4, 1.0, 0.0, 1.0],
///     vec![1.0, 1.4, 1.0, 0.0],
/// ];
/// let diagrams = compute_persistence(&dist, 1, 2.0).expect("should succeed");
/// assert_eq!(diagrams.len(), 2); // H0 and H1
/// assert!(!diagrams[0].is_empty()); // at least one H0 feature
/// ```
pub fn compute_persistence(
    distance_matrix: &[Vec<f64>],
    max_dim: usize,
    max_epsilon: f64,
) -> Result<Vec<PersistenceDiagram>> {
    let n = distance_matrix.len();
    if n == 0 {
        // Return empty diagrams
        return Ok((0..=max_dim).map(|d| PersistenceDiagram::new(d)).collect());
    }

    // Validate distance matrix
    for row in distance_matrix {
        if row.len() != n {
            return Err(TransformError::InvalidInput(
                "distance_matrix must be square".to_string(),
            ));
        }
    }
    if max_epsilon < 0.0 {
        return Err(TransformError::InvalidInput(
            "max_epsilon must be non-negative".to_string(),
        ));
    }

    // Build points from distance matrix for the VietorisRips struct
    // We use the ndarray interface that VietorisRips::compute expects.
    // Convert the distance matrix to an Array2 of positions via MDS-like embedding
    // (actually VietorisRips::compute accepts the data matrix, not distance).
    // We use the approach of lifting points into n-dimensional space via
    // the distance matrix's row vectors (approximate MDS, sufficient for filtration).
    //
    // However since VietorisRips::compute takes a data matrix and recomputes
    // Euclidean distances, we need to provide point coordinates such that
    // ||p_i - p_j|| = distance_matrix[i][j].
    //
    // For the general case we use the distance matrix directly to construct
    // the filtration manually (boundary matrix reduction).

    // Collect all unique pairwise distances (filtration values)
    let mut filt_values: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance_matrix[i][j];
            if d <= max_epsilon && d >= 0.0 {
                filt_values.push(d);
            }
        }
    }
    filt_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    filt_values.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Enumerate simplices with their filtration values
    // 0-simplices: all vertices, filtration 0.0
    // 1-simplices: edges, filtration = dist(i,j)
    // 2-simplices: triangles, filtration = max edge length

    #[derive(Clone)]
    struct FiltSimplex {
        vertices: Vec<usize>,
        filtration: f64,
    }

    let mut simplices: Vec<FiltSimplex> = Vec::new();

    // Vertices
    for i in 0..n {
        simplices.push(FiltSimplex {
            vertices: vec![i],
            filtration: 0.0,
        });
    }

    // Edges
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance_matrix[i][j];
            if d <= max_epsilon {
                simplices.push(FiltSimplex {
                    vertices: vec![i, j],
                    filtration: d,
                });
            }
        }
    }

    // Triangles (for H1 / H2)
    if max_dim >= 1 {
        for i in 0..n {
            for j in (i + 1)..n {
                let d_ij = distance_matrix[i][j];
                if d_ij > max_epsilon {
                    continue;
                }
                for k in (j + 1)..n {
                    let d_ik = distance_matrix[i][k];
                    let d_jk = distance_matrix[j][k];
                    if d_ik > max_epsilon || d_jk > max_epsilon {
                        continue;
                    }
                    let max_d = d_ij.max(d_ik).max(d_jk);
                    simplices.push(FiltSimplex {
                        vertices: vec![i, j, k],
                        filtration: max_d,
                    });
                }
            }
        }
    }

    // Tetrahedra (for H2 / H3)
    if max_dim >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                let d_ij = distance_matrix[i][j];
                if d_ij > max_epsilon {
                    continue;
                }
                for k in (j + 1)..n {
                    let d_ik = distance_matrix[i][k];
                    let d_jk = distance_matrix[j][k];
                    if d_ik > max_epsilon || d_jk > max_epsilon {
                        continue;
                    }
                    for l in (k + 1)..n {
                        let d_il = distance_matrix[i][l];
                        let d_jl = distance_matrix[j][l];
                        let d_kl = distance_matrix[k][l];
                        if d_il > max_epsilon || d_jl > max_epsilon || d_kl > max_epsilon {
                            continue;
                        }
                        let max_d = d_ij.max(d_ik).max(d_jk).max(d_il).max(d_jl).max(d_kl);
                        simplices.push(FiltSimplex {
                            vertices: vec![i, j, k, l],
                            filtration: max_d,
                        });
                    }
                }
            }
        }
    }

    // Sort by filtration value, then by dimension
    simplices.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.vertices.len().cmp(&b.vertices.len()))
    });

    // Index simplices for boundary matrix
    let total = simplices.len();
    let simplex_idx: std::collections::HashMap<Vec<usize>, usize> = simplices
        .iter()
        .enumerate()
        .map(|(i, s)| (s.vertices.clone(), i))
        .collect();

    // Build boundary matrix columns as sorted lists of row indices (mod 2)
    // col j = list of (index of) (dim-1)-faces of simplex j
    let mut boundary: Vec<Vec<usize>> = vec![Vec::new(); total];
    for (j, simp) in simplices.iter().enumerate() {
        let d = simp.vertices.len();
        if d <= 1 {
            continue; // 0-simplex has empty boundary
        }
        // Faces: remove one vertex at a time
        for omit in 0..d {
            let face: Vec<usize> = simp
                .vertices
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != omit)
                .map(|(_, &v)| v)
                .collect();
            if let Some(&row_idx) = simplex_idx.get(&face) {
                boundary[j].push(row_idx);
            }
        }
        boundary[j].sort_unstable();
    }

    // Persistence pairing via standard reduction (column reduction over F_2)
    // low[j] = lowest row index in column j (None if zero column)
    let mut low: Vec<Option<usize>> = vec![None; total];
    // pivot_col[r] = column that has lowest row index r
    let mut pivot_col: Vec<Option<usize>> = vec![None; total];

    for j in 0..total {
        loop {
            let lo = boundary[j].last().copied();
            match lo {
                None => break,
                Some(r) => {
                    if let Some(k) = pivot_col[r] {
                        // Add column k to column j (mod 2)
                        let bk = boundary[k].clone();
                        sym_diff_inplace(&mut boundary[j], &bk);
                    } else {
                        low[j] = Some(r);
                        pivot_col[r] = Some(j);
                        break;
                    }
                }
            }
        }
    }

    // Extract persistence pairs
    let mut diagrams: Vec<PersistenceDiagram> =
        (0..=max_dim).map(|d| PersistenceDiagram::new(d)).collect();

    let mut paired: Vec<bool> = vec![false; total];

    for j in 0..total {
        if let Some(r) = low[j] {
            let birth = simplices[r].filtration;
            let death = simplices[j].filtration;
            // The dimension of the feature is dim(simplex r) = r's vertex count - 1
            let feature_dim = simplices[r].vertices.len() - 1;
            if feature_dim <= max_dim && (death - birth).abs() > 1e-15 {
                diagrams[feature_dim].add_point(birth, death, feature_dim);
            }
            paired[r] = true;
            paired[j] = true;
        }
    }

    // Unpaired simplices → essential features
    for i in 0..total {
        if !paired[i] {
            let dim = simplices[i].vertices.len() - 1;
            if dim <= max_dim {
                diagrams[dim].add_point(simplices[i].filtration, f64::INFINITY, dim);
            }
        }
    }

    Ok(diagrams)
}

// ─── persistence_landscape free function ─────────────────────────────────────

/// Compute the persistence landscape of a diagram, evaluated at points `x`.
///
/// The persistence landscape λ_k(t) is defined as:
///   λ_k(t) = kth largest value of  min(t - b, d - t)⁺  over all (b, d) pairs.
///
/// # Arguments
/// * `dgm`      — persistence diagram (only finite points are used)
/// * `n_layers` — number of landscape layers to compute
/// * `x`        — evaluation points (must be sorted in ascending order)
///
/// # Returns
/// Matrix of shape (n_layers x len(x)), where entry `[k, i]` = lambda\_{k+1}(x\[i\]).
///
/// # Example
///
/// ```rust
/// use scirs2_transform::tda_ext::{compute_persistence, persistence_landscape_fn};
///
/// let dist = vec![
///     vec![0.0, 1.0, 1.4, 1.0],
///     vec![1.0, 0.0, 1.0, 1.4],
///     vec![1.4, 1.0, 0.0, 1.0],
///     vec![1.0, 1.4, 1.0, 0.0],
/// ];
/// let diagrams = compute_persistence(&dist, 1, 2.0).expect("should succeed");
/// let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
/// let landscape = persistence_landscape_fn(&diagrams[0], 2, &x);
/// assert_eq!(landscape.len(), 2);       // n_layers
/// assert_eq!(landscape[0].len(), 20);   // len(x)
/// ```
pub fn persistence_landscape_fn(
    dgm: &PersistenceDiagram,
    n_layers: usize,
    x: &[f64],
) -> Vec<Vec<f64>> {
    if n_layers == 0 || x.is_empty() {
        return vec![vec![0.0; x.len()]; n_layers];
    }

    // Collect finite (b, d) pairs
    let finite_pts: Vec<(f64, f64)> = dgm
        .points
        .iter()
        .filter(|p| p.death.is_finite())
        .map(|p| (p.birth, p.death))
        .collect();

    let nx = x.len();
    // For each evaluation point t, compute tent values and keep top n_layers
    let mut landscape = vec![vec![0.0_f64; nx]; n_layers];

    for (ix, &t) in x.iter().enumerate() {
        // Tent function value for each pair at t
        let mut tents: Vec<f64> = finite_pts
            .iter()
            .map(|&(b, d)| {
                let v = (t - b).min(d - t);
                if v < 0.0 {
                    0.0
                } else {
                    v
                }
            })
            .collect();
        // Sort descending and take top n_layers
        tents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for k in 0..n_layers {
            landscape[k][ix] = tents.get(k).copied().unwrap_or(0.0);
        }
    }

    landscape
}

// ─── persistence_image free function ─────────────────────────────────────────

/// Compute a persistence image from a persistence diagram.
///
/// Maps each point (b, p) (where p = d − b is persistence) in the diagram to
/// a 2D Gaussian kernel, then evaluates on a regular grid over
/// [0, max_birth] × [0, max_persistence].
///
/// Points are weighted by their persistence (linear weighting function).
///
/// # Arguments
/// * `dgm`             — source persistence diagram
/// * `bandwidth`       — Gaussian kernel bandwidth (σ)
/// * `grid`            — (n_rows, n_cols) resolution of the output image
/// * `max_birth`       — upper bound of the birth axis
/// * `max_persistence` — upper bound of the persistence axis
///
/// # Returns
/// 2D grid of shape (n_rows × n_cols) as a Vec of rows.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::tda_ext::{compute_persistence, persistence_image_fn};
///
/// let dist = vec![
///     vec![0.0, 1.0, 1.4, 1.0],
///     vec![1.0, 0.0, 1.0, 1.4],
///     vec![1.4, 1.0, 0.0, 1.0],
///     vec![1.0, 1.4, 1.0, 0.0],
/// ];
/// let diagrams = compute_persistence(&dist, 0, 2.0).expect("should succeed");
/// let img = persistence_image_fn(&diagrams[0], 0.1, (5, 5), 2.0, 2.0);
/// assert_eq!(img.len(), 5);
/// assert_eq!(img[0].len(), 5);
/// ```
pub fn persistence_image_fn(
    dgm: &PersistenceDiagram,
    bandwidth: f64,
    grid: (usize, usize),
    max_birth: f64,
    max_persistence: f64,
) -> Vec<Vec<f64>> {
    let (n_rows, n_cols) = grid;
    if n_rows == 0 || n_cols == 0 {
        return vec![];
    }

    let bw = bandwidth.max(1e-10);
    let two_bw_sq = 2.0 * bw * bw;
    let norm_factor = 1.0 / (std::f64::consts::TAU * bw * bw);

    // Grid cell centres
    let row_centers: Vec<f64> = if n_rows == 1 {
        vec![max_persistence * 0.5]
    } else {
        (0..n_rows)
            .map(|i| max_persistence * i as f64 / (n_rows - 1) as f64)
            .collect()
    };
    let col_centers: Vec<f64> = if n_cols == 1 {
        vec![max_birth * 0.5]
    } else {
        (0..n_cols)
            .map(|j| max_birth * j as f64 / (n_cols - 1) as f64)
            .collect()
    };

    // Collect finite (birth, persistence) pairs with weight = persistence
    let pts: Vec<(f64, f64, f64)> = dgm
        .points
        .iter()
        .filter(|p| p.death.is_finite() && p.death > p.birth)
        .map(|p| (p.birth, p.death - p.birth, p.death - p.birth)) // (b, pers, weight)
        .collect();

    let mut image = vec![vec![0.0_f64; n_cols]; n_rows];

    for (r, &p_center) in row_centers.iter().enumerate() {
        for (c, &b_center) in col_centers.iter().enumerate() {
            let mut val = 0.0_f64;
            for &(b, pers, weight) in &pts {
                let db = b_center - b;
                let dp = p_center - pers;
                let exponent = -(db * db + dp * dp) / two_bw_sq;
                val += weight * norm_factor * exponent.exp();
            }
            image[r][c] = val;
        }
    }

    image
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute n×n pairwise Euclidean distances.
fn pairwise_distances(points: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0_f64;
            for d in 0..dim.min(points[i].len()).min(points[j].len()) {
                let diff = points[i][d] - points[j][d];
                sq += diff * diff;
            }
            let d = sq.sqrt();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Symmetric difference of two sorted Vec<usize> in-place (mod-2 addition of boundary chains).
fn sym_diff_inplace(a: &mut Vec<usize>, b: &[usize]) {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut ai = 0_usize;
    let mut bi = 0_usize;
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => {
                result.push(a[ai]);
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[bi]);
                bi += 1;
            }
            std::cmp::Ordering::Equal => {
                // Cancel (mod 2): skip both
                ai += 1;
                bi += 1;
            }
        }
    }
    while ai < a.len() {
        result.push(a[ai]);
        ai += 1;
    }
    while bi < b.len() {
        result.push(b[bi]);
        bi += 1;
    }
    *a = result;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tda_vr::PersistenceDiagram;

    fn square_dist() -> Vec<Vec<f64>> {
        // 4 points: (0,0), (1,0), (1,1), (0,1) — unit square
        vec![
            vec![0.0, 1.0, 1.414, 1.0],
            vec![1.0, 0.0, 1.0, 1.414],
            vec![1.414, 1.0, 0.0, 1.0],
            vec![1.0, 1.414, 1.0, 0.0],
        ]
    }

    fn square_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ]
    }

    // ── VietorisRipsComplex ───────────────────────────────────────────────────

    #[test]
    fn test_vrc_vertices() {
        let pts = square_points();
        let vrc = VietorisRipsComplex::new(&pts, 1.5).expect("new");
        assert_eq!(vrc.n_simplices(0), 4, "Should have 4 vertices");
    }

    #[test]
    fn test_vrc_edges_unit_square() {
        let pts = square_points();
        // At epsilon = 1.0: only 4 edges (sides), no diagonals (sqrt(2) ≈ 1.414 > 1.0)
        let vrc = VietorisRipsComplex::new(&pts, 1.0).expect("new");
        assert_eq!(vrc.n_simplices(1), 4, "Unit square at eps=1 has 4 edges");
        assert_eq!(vrc.n_simplices(2), 0, "No triangles at eps=1");
    }

    #[test]
    fn test_vrc_complete_graph() {
        let pts = square_points();
        // At epsilon = 2.0: all 6 edges present → 4 triangles
        let vrc = VietorisRipsComplex::new(&pts, 2.0).expect("new");
        assert_eq!(
            vrc.n_simplices(1),
            6,
            "Complete graph on 4 vertices has 6 edges"
        );
        assert_eq!(vrc.n_simplices(2), 4, "4 triangles in K4");
    }

    #[test]
    fn test_vrc_euler_characteristic() {
        let pts = square_points();
        // At eps = 1.0: 4 vertices, 4 edges, 0 triangles → χ = 4 - 4 + 0 = 0
        let vrc = VietorisRipsComplex::new(&pts, 1.0).expect("new");
        assert_eq!(vrc.euler_characteristic(), 0);
    }

    #[test]
    fn test_vrc_empty_input() {
        let vrc = VietorisRipsComplex::new(&[], 1.0).expect("empty ok");
        assert_eq!(vrc.n_simplices(0), 0);
        assert_eq!(vrc.euler_characteristic(), 0);
    }

    #[test]
    fn test_vrc_negative_epsilon_error() {
        let pts = square_points();
        assert!(VietorisRipsComplex::new(&pts, -0.1).is_err());
    }

    #[test]
    fn test_vrc_are_connected() {
        let pts = square_points();
        let vrc = VietorisRipsComplex::new(&pts, 1.0).expect("new");
        // Edges of the square: (0,1), (1,2), (2,3), (0,3)
        assert!(vrc.are_connected(0, 1));
        assert!(vrc.are_connected(1, 2));
        // Diagonal (0,2) has length sqrt(2) > 1.0
        assert!(!vrc.are_connected(0, 2));
    }

    // ── compute_persistence ───────────────────────────────────────────────────

    #[test]
    fn test_compute_persistence_h0_square() {
        let dist = square_dist();
        let diagrams = compute_persistence(&dist, 1, 2.0).expect("persistence");
        assert_eq!(diagrams.len(), 2);
        let h0 = &diagrams[0];
        // Should have H0 features (at least one)
        assert!(!h0.is_empty(), "H0 should not be empty");
    }

    #[test]
    fn test_compute_persistence_empty() {
        let diagrams = compute_persistence(&[], 1, 1.0).expect("empty");
        assert_eq!(diagrams.len(), 2); // H0 and H1, both empty
        assert!(diagrams[0].is_empty());
        assert!(diagrams[1].is_empty());
    }

    #[test]
    fn test_compute_persistence_non_square_error() {
        let dist = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]];
        assert!(compute_persistence(&dist, 1, 2.0).is_err());
    }

    #[test]
    fn test_compute_persistence_returns_finite_pairs() {
        let dist = square_dist();
        let diagrams = compute_persistence(&dist, 1, 2.0).expect("persistence");
        for dgm in &diagrams {
            for pt in &dgm.points {
                assert!(pt.birth.is_finite());
                assert!(pt.birth >= 0.0);
                if pt.death.is_finite() {
                    assert!(pt.death >= pt.birth);
                }
            }
        }
    }

    // ── persistence_landscape_fn ──────────────────────────────────────────────

    #[test]
    fn test_landscape_fn_shape() {
        let mut dgm = PersistenceDiagram::new(0);
        dgm.add_point(0.0, 2.0, 0);
        dgm.add_point(0.5, 1.5, 0);
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let l = persistence_landscape_fn(&dgm, 3, &x);
        assert_eq!(l.len(), 3);
        assert_eq!(l[0].len(), 20);
    }

    #[test]
    fn test_landscape_fn_non_negative() {
        let mut dgm = PersistenceDiagram::new(0);
        dgm.add_point(0.0, 1.0, 0);
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.15).collect();
        let l = persistence_landscape_fn(&dgm, 2, &x);
        for row in &l {
            for &v in row {
                assert!(v >= 0.0, "landscape must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn test_landscape_fn_tent_shape() {
        let mut dgm = PersistenceDiagram::new(0);
        // Single (0, 1) pair → tent peaked at t=0.5 with height 0.5
        dgm.add_point(0.0, 1.0, 0);
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let l = persistence_landscape_fn(&dgm, 1, &x);
        // λ_1(0.5) = min(0.5 - 0, 1.0 - 0.5) = 0.5
        assert!((l[0][2] - 0.5).abs() < 1e-10, "peak should be 0.5");
        // λ_1(0) = 0, λ_1(1) = 0
        assert!(l[0][0] < 1e-10);
        assert!(l[0][4] < 1e-10);
    }

    #[test]
    fn test_landscape_fn_empty_diagram() {
        let dgm = PersistenceDiagram::new(0);
        let x = vec![0.0, 1.0, 2.0];
        let l = persistence_landscape_fn(&dgm, 2, &x);
        assert_eq!(l.len(), 2);
        for row in &l {
            assert!(row.iter().all(|&v| v == 0.0));
        }
    }

    // ── persistence_image_fn ──────────────────────────────────────────────────

    #[test]
    fn test_persistence_image_fn_shape() {
        let mut dgm = PersistenceDiagram::new(0);
        dgm.add_point(0.0, 1.0, 0);
        dgm.add_point(0.2, 0.8, 0);
        let img = persistence_image_fn(&dgm, 0.1, (5, 5), 1.0, 1.0);
        assert_eq!(img.len(), 5);
        assert_eq!(img[0].len(), 5);
    }

    #[test]
    fn test_persistence_image_fn_non_negative() {
        let mut dgm = PersistenceDiagram::new(0);
        dgm.add_point(0.0, 1.0, 0);
        let img = persistence_image_fn(&dgm, 0.1, (4, 4), 1.0, 1.0);
        for row in &img {
            for &v in row {
                assert!(v >= 0.0, "image pixel must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn test_persistence_image_fn_has_signal() {
        let mut dgm = PersistenceDiagram::new(0);
        dgm.add_point(0.0, 1.0, 0);
        let img = persistence_image_fn(&dgm, 0.15, (6, 6), 1.5, 1.5);
        let has_positive = img.iter().flat_map(|row| row.iter()).any(|&v| v > 0.0);
        assert!(
            has_positive,
            "image should have nonzero pixels for a nonempty diagram"
        );
    }

    #[test]
    fn test_persistence_image_fn_empty_diagram() {
        let dgm = PersistenceDiagram::new(0);
        let img = persistence_image_fn(&dgm, 0.1, (4, 4), 1.0, 1.0);
        assert_eq!(img.len(), 4);
        for row in &img {
            assert!(row.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_sym_diff_inplace() {
        let mut a = vec![1_usize, 3, 5];
        let b = vec![2, 3, 4];
        sym_diff_inplace(&mut a, &b);
        // Expected: 1, 2, 4, 5 (3 cancelled)
        assert_eq!(a, vec![1, 2, 4, 5]);
    }
}
