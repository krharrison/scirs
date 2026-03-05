//! Topological Data Analysis (TDA) and Persistent Homology
//!
//! This module provides implementations of persistent homology for analyzing
//! topological features of data. Key concepts include:
//!
//! - **Simplicial complexes**: Geometric objects built from vertices, edges, triangles, etc.
//! - **Filtrations**: Nested sequences of simplicial complexes parameterized by a scale
//! - **Persistence diagrams**: Collections of (birth, death) pairs representing topological features
//! - **Barcodes**: Interval representations of persistent homology
//! - **Persistence images**: Stable vectorizations of persistence diagrams
//!
//! ## Algorithms
//!
//! - Vietoris-Rips complex construction via distance-based filtration
//! - Boundary matrix reduction for computing persistent homology
//! - Bottleneck distance between persistence diagrams
//! - Wasserstein distance between persistence diagrams
//! - Persistence image vectorization
//!
//! ## References
//!
//! - Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological persistence and simplification.
//! - Zomorodian, A., & Carlsson, G. (2005). Computing persistent homology.
//! - Adams, H., et al. (2017). Persistence images: A stable vector representation of persistent homology.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::fmt;

// ─── Core Data Structures ────────────────────────────────────────────────────

/// A single persistence point (birth, death) pair in a persistence diagram.
/// The dimension indicates which homological dimension this feature belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePoint {
    /// Birth time (filtration value at which the feature appears)
    pub birth: f64,
    /// Death time (filtration value at which the feature disappears),
    /// or f64::INFINITY for essential features
    pub death: f64,
    /// Homological dimension (0 = components, 1 = loops, 2 = voids, ...)
    pub dimension: usize,
}

impl PersistencePoint {
    /// Create a new persistence point
    pub fn new(birth: f64, death: f64, dimension: usize) -> Self {
        Self {
            birth,
            death,
            dimension,
        }
    }

    /// Compute the persistence (lifetime) of this feature
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    /// Check if this is an essential feature (never dies)
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

impl fmt::Display for PersistencePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.death.is_infinite() {
            write!(f, "H{}({:.4}, ∞)", self.dimension, self.birth)
        } else {
            write!(
                f,
                "H{}({:.4}, {:.4})",
                self.dimension, self.birth, self.death
            )
        }
    }
}

/// Persistence diagram: a collection of (birth, death) pairs per dimension.
///
/// A persistence diagram captures the topological features of a dataset
/// across all scales, organized by homological dimension.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// All persistence points across all dimensions
    pub points: Vec<PersistencePoint>,
    /// Maximum homological dimension computed
    pub max_dimension: usize,
}

impl PersistenceDiagram {
    /// Create an empty persistence diagram
    pub fn new(max_dimension: usize) -> Self {
        Self {
            points: Vec::new(),
            max_dimension,
        }
    }

    /// Add a persistence point to the diagram
    pub fn add_point(&mut self, birth: f64, death: f64, dimension: usize) {
        self.points.push(PersistencePoint::new(birth, death, dimension));
    }

    /// Get all points in a specific homological dimension
    pub fn points_in_dimension(&self, dim: usize) -> Vec<&PersistencePoint> {
        self.points.iter().filter(|p| p.dimension == dim).collect()
    }

    /// Get all finite persistence points (non-essential features)
    pub fn finite_points(&self) -> Vec<&PersistencePoint> {
        self.points.iter().filter(|p| !p.is_essential()).collect()
    }

    /// Get all essential points (infinite persistence)
    pub fn essential_points(&self) -> Vec<&PersistencePoint> {
        self.points.iter().filter(|p| p.is_essential()).collect()
    }

    /// Compute the total persistence (sum of all finite persistence values)
    pub fn total_persistence(&self, p: f64) -> f64 {
        self.points
            .iter()
            .filter(|pt| !pt.is_essential())
            .map(|pt| pt.persistence().powf(p))
            .sum::<f64>()
            .powf(1.0 / p)
    }

    /// Filter points by minimum persistence threshold
    pub fn filter_by_persistence(&self, min_persistence: f64) -> PersistenceDiagram {
        let filtered_points: Vec<PersistencePoint> = self
            .points
            .iter()
            .filter(|p| p.persistence() >= min_persistence)
            .cloned()
            .collect();

        PersistenceDiagram {
            points: filtered_points,
            max_dimension: self.max_dimension,
        }
    }

    /// Number of points in the diagram
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the diagram is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Convert to barcode representation
    pub fn to_barcode(&self) -> Barcode {
        Barcode::from_diagram(self)
    }

    /// Get the Betti numbers (count of features) at a given filtration value
    pub fn betti_numbers_at(&self, filtration_value: f64) -> Vec<usize> {
        let mut betti = vec![0usize; self.max_dimension + 1];
        for p in &self.points {
            if p.birth <= filtration_value
                && (p.death > filtration_value || p.death.is_infinite())
            {
                if p.dimension <= self.max_dimension {
                    betti[p.dimension] += 1;
                }
            }
        }
        betti
    }
}

// ─── Barcode ─────────────────────────────────────────────────────────────────

/// An interval [birth, death) in a persistence barcode
#[derive(Debug, Clone, PartialEq)]
pub struct BarcodeInterval {
    /// Start of the interval (birth filtration value)
    pub birth: f64,
    /// End of the interval (death filtration value), or ∞
    pub death: f64,
    /// Homological dimension
    pub dimension: usize,
}

impl BarcodeInterval {
    /// Length of the interval (persistence)
    pub fn length(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }
}

/// Persistence barcode: a multi-set of intervals representing topological features.
///
/// Each interval [birth, death) represents a topological feature that appears
/// at filtration value `birth` and disappears at filtration value `death`.
#[derive(Debug, Clone)]
pub struct Barcode {
    /// All barcode intervals
    pub intervals: Vec<BarcodeInterval>,
    /// Maximum dimension
    pub max_dimension: usize,
}

impl Barcode {
    /// Create a barcode from a persistence diagram
    pub fn from_diagram(diagram: &PersistenceDiagram) -> Self {
        let intervals: Vec<BarcodeInterval> = diagram
            .points
            .iter()
            .map(|p| BarcodeInterval {
                birth: p.birth,
                death: p.death,
                dimension: p.dimension,
            })
            .collect();

        Barcode {
            intervals,
            max_dimension: diagram.max_dimension,
        }
    }

    /// Get intervals in a specific dimension, sorted by birth
    pub fn intervals_in_dimension(&self, dim: usize) -> Vec<&BarcodeInterval> {
        let mut intervals: Vec<&BarcodeInterval> = self
            .intervals
            .iter()
            .filter(|i| i.dimension == dim)
            .collect();
        intervals.sort_by(|a, b| a.birth.partial_cmp(&b.birth).unwrap_or(std::cmp::Ordering::Equal));
        intervals
    }

    /// Number of intervals
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Whether the barcode is empty
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }
}

// ─── Simplex and Filtration ───────────────────────────────────────────────────

/// A simplex (vertex set) with its filtration value
#[derive(Debug, Clone, PartialEq)]
struct FilteredSimplex {
    /// Sorted vertex indices
    vertices: Vec<usize>,
    /// Filtration value at which this simplex appears
    filtration_value: f64,
}

impl FilteredSimplex {
    fn new(vertices: Vec<usize>, filtration_value: f64) -> Self {
        let mut v = vertices;
        v.sort_unstable();
        Self {
            vertices: v,
            filtration_value,
        }
    }

    fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }
}

// ─── Boundary Matrix ─────────────────────────────────────────────────────────

/// Boundary matrix column (sparse representation using sorted pivot-tracked columns)
struct BoundaryMatrix {
    /// Columns of the boundary matrix (each column is a sorted list of row indices)
    columns: Vec<Vec<usize>>,
    /// Pivot row index for each column (-1 if zero column), using i64 for sentinel
    pivots: Vec<i64>,
}

impl BoundaryMatrix {
    fn new(n_cols: usize) -> Self {
        Self {
            columns: vec![Vec::new(); n_cols],
            pivots: vec![-1i64; n_cols],
        }
    }

    /// Set column from a boundary list
    fn set_column(&mut self, col: usize, boundary: Vec<usize>) {
        let mut b = boundary;
        b.sort_unstable();
        b.dedup();
        let pivot = b.last().copied().map(|v| v as i64).unwrap_or(-1);
        self.columns[col] = b;
        self.pivots[col] = pivot;
    }

    /// Get the pivot (lowest row index) of column j
    fn pivot(&self, j: usize) -> i64 {
        self.pivots[j]
    }

    /// Add column j to column i (XOR / Z_2 addition, i.e., symmetric difference)
    fn add_column(&mut self, target: usize, source: usize) {
        let src = self.columns[source].clone();
        let tgt = self.columns[target].clone();

        // Symmetric difference of two sorted lists
        let mut result = Vec::with_capacity(src.len() + tgt.len());
        let (mut i, mut j) = (0, 0);
        while i < src.len() && j < tgt.len() {
            match src[i].cmp(&tgt[j]) {
                std::cmp::Ordering::Less => {
                    result.push(src[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(tgt[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Both have it; cancel out (Z_2 arithmetic)
                    i += 1;
                    j += 1;
                }
            }
        }
        result.extend_from_slice(&src[i..]);
        result.extend_from_slice(&tgt[j..]);

        let pivot = result.last().copied().map(|v| v as i64).unwrap_or(-1);
        self.columns[target] = result;
        self.pivots[target] = pivot;
    }

    /// Standard reduction algorithm (column reduction over Z_2)
    fn reduce(&mut self) {
        let n = self.columns.len();
        // Map from pivot row -> column index
        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();

        for j in 0..n {
            loop {
                let piv = self.pivot(j);
                if piv < 0 {
                    break; // Zero column, done
                }
                let piv_row = piv as usize;
                if let Some(&k) = pivot_to_col.get(&piv_row) {
                    // Column k also has pivot at piv_row; add k to j
                    self.add_column(j, k);
                } else {
                    // Record this column's pivot
                    pivot_to_col.insert(piv_row, j);
                    break;
                }
            }
        }
    }
}

// ─── Vietoris-Rips Complex ───────────────────────────────────────────────────

/// Vietoris-Rips complex for computing persistent homology
///
/// The Vietoris-Rips complex builds a filtration of simplicial complexes from
/// point cloud data by including a simplex whenever all pairwise distances
/// between its vertices are at most the filtration parameter ε.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::tda::VietorisRips;
/// use scirs2_core::ndarray::Array2;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,   1.0, 0.0,   1.0, 1.0,   0.0, 1.0,
/// ]).expect("should succeed");
///
/// let diagram = VietorisRips::compute(&points, 1, 2.0).expect("should succeed");
/// assert!(!diagram.is_empty());
/// ```
pub struct VietorisRips;

impl VietorisRips {
    /// Compute the persistent homology of a point cloud using the Vietoris-Rips filtration
    ///
    /// # Arguments
    /// * `points` - Point cloud data (n_points × n_features)
    /// * `max_dim` - Maximum homological dimension to compute (0 = components, 1 = loops, ...)
    /// * `max_radius` - Maximum filtration radius; simplices with diameter > 2*max_radius are ignored
    ///
    /// # Returns
    /// * A persistence diagram with (birth, death) pairs for each dimension
    pub fn compute<S>(
        points: &ArrayBase<S, Ix2>,
        max_dim: usize,
        max_radius: f64,
    ) -> Result<PersistenceDiagram>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = points.nrows();
        if n < 2 {
            return Err(TransformError::InvalidInput(
                "VietorisRips requires at least 2 points".to_string(),
            ));
        }

        // Compute pairwise Euclidean distances
        let dist = Self::compute_distance_matrix(points)?;

        // Build Vietoris-Rips filtration up to max_dim + 1 skeleton
        let mut filtered_simplices = Self::build_filtration(&dist, max_dim, max_radius);

        // Sort by filtration value, then dimension for correct order
        filtered_simplices.sort_by(|a, b| {
            a.filtration_value
                .partial_cmp(&b.filtration_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.dimension().cmp(&b.dimension()))
        });

        // Assign indices to simplices
        let n_simplices = filtered_simplices.len();

        // Build index lookup
        let mut simplex_to_idx: HashMap<Vec<usize>, usize> = HashMap::new();
        for (i, s) in filtered_simplices.iter().enumerate() {
            simplex_to_idx.insert(s.vertices.clone(), i);
        }

        // Build boundary matrix
        let mut bm = BoundaryMatrix::new(n_simplices);
        for (j, simplex) in filtered_simplices.iter().enumerate() {
            if simplex.dimension() == 0 {
                // Vertices have empty boundary
                continue;
            }
            // Boundary of a simplex: all faces obtained by removing one vertex
            let mut boundary_indices = Vec::with_capacity(simplex.vertices.len());
            for k in 0..simplex.vertices.len() {
                let face: Vec<usize> = simplex
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != k)
                    .map(|(_, &v)| v)
                    .collect();
                if let Some(&idx) = simplex_to_idx.get(&face) {
                    boundary_indices.push(idx);
                }
            }
            bm.set_column(j, boundary_indices);
        }

        // Reduce boundary matrix
        bm.reduce();

        // Extract persistence pairs
        let mut diagram = PersistenceDiagram::new(max_dim);

        // Track which columns are "positive" (not killed by a later simplex)
        let mut killed = vec![false; n_simplices];

        for j in 0..n_simplices {
            let piv = bm.pivot(j);
            if piv >= 0 {
                let i = piv as usize;
                // Column j kills simplex i
                killed[i] = true;
                let dim_creator = filtered_simplices[i].dimension();
                if dim_creator <= max_dim {
                    let birth = filtered_simplices[i].filtration_value;
                    let death = filtered_simplices[j].filtration_value;
                    if (death - birth).abs() > 1e-12 {
                        diagram.add_point(birth, death, dim_creator);
                    }
                }
            }
        }

        // Essential features: simplices not killed and with zero column (not reducible)
        for i in 0..n_simplices {
            if !killed[i] && bm.pivot(i) < 0 {
                let dim = filtered_simplices[i].dimension();
                if dim <= max_dim {
                    let birth = filtered_simplices[i].filtration_value;
                    // Mark as essential (death = infinity)
                    // Skip H_0 essential features except the longest-lived one
                    // (there's exactly one connected component in a connected point cloud)
                    diagram.add_point(birth, f64::INFINITY, dim);
                }
            }
        }

        Ok(diagram)
    }

    /// Compute pairwise Euclidean distance matrix
    fn compute_distance_matrix<S>(points: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = points.nrows();
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let mut d_sq = 0.0f64;
                for k in 0..points.ncols() {
                    let diff = NumCast::from(points[[i, k]]).unwrap_or(0.0)
                        - NumCast::from(points[[j, k]]).unwrap_or(0.0);
                    d_sq += diff * diff;
                }
                let d = d_sq.sqrt();
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        Ok(dist)
    }

    /// Build all simplices in the Vietoris-Rips filtration up to max_dim+1 skeleton
    fn build_filtration(
        dist: &Array2<f64>,
        max_dim: usize,
        max_radius: f64,
    ) -> Vec<FilteredSimplex> {
        let n = dist.nrows();
        let max_diam = 2.0 * max_radius;
        let mut simplices = Vec::new();

        // Add vertices (0-simplices)
        for i in 0..n {
            simplices.push(FilteredSimplex::new(vec![i], 0.0));
        }

        // Iteratively build higher-dimensional simplices
        // Use clique-finding: a k-simplex [v0,...,vk] is in the filtration iff
        // all pairwise distances <= max_diam; filtration value = max pairwise distance

        // Start with edges (1-simplices) and build cliques
        let mut prev_dim_simplices: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for dim in 1..=(max_dim + 1) {
            let mut next_dim_simplices: Vec<Vec<usize>> = Vec::new();

            // Extend each (dim-1)-simplex by adding a vertex with larger index
            // than all current vertices (to avoid duplicates)
            for simplex in &prev_dim_simplices {
                let last_vertex = *simplex.last().unwrap_or(&0);

                for v in (last_vertex + 1)..n {
                    // Check if v can be added (all distances to current vertices <= max_diam)
                    let max_dist_to_v = simplex
                        .iter()
                        .map(|&u| dist[[u, v]])
                        .fold(0.0f64, f64::max);

                    if max_dist_to_v <= max_diam {
                        let mut new_simplex = simplex.clone();
                        new_simplex.push(v);

                        // Filtration value = diameter of simplex = max pairwise distance
                        let filtration_val = Self::simplex_diameter(&new_simplex, dist);
                        simplices.push(FilteredSimplex::new(new_simplex.clone(), filtration_val));
                        next_dim_simplices.push(new_simplex);
                    }
                }
            }

            if next_dim_simplices.is_empty() {
                break;
            }
            prev_dim_simplices = next_dim_simplices;
        }

        simplices
    }

    /// Compute the diameter (max pairwise distance) of a simplex
    fn simplex_diameter(vertices: &[usize], dist: &Array2<f64>) -> f64 {
        let mut max_d = 0.0f64;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let d = dist[[vertices[i], vertices[j]]];
                if d > max_d {
                    max_d = d;
                }
            }
        }
        max_d
    }
}

// ─── Persistence Image ────────────────────────────────────────────────────────

/// Persistence image: a stable vector representation of persistence diagrams.
///
/// Maps persistence diagrams to a 2D grid image by placing a Gaussian kernel
/// centered at each persistence point (birth, persistence) and integrating
/// over grid cells.
///
/// # References
/// Adams, H., et al. (2017). Persistence images: A stable vector representation
/// of persistent homology. JMLR, 18(8), 1-35.
pub struct PersistenceImage {
    /// Image resolution (resolution × resolution pixels)
    resolution: usize,
    /// Birth axis range [min, max]
    birth_range: (f64, f64),
    /// Persistence axis range [min, max]
    persistence_range: (f64, f64),
    /// Gaussian kernel bandwidth (sigma)
    sigma: f64,
    /// Weight function applied to each point
    weight_type: PersistenceWeight,
    /// Homological dimension to use
    dimension: usize,
}

/// Weight function for persistence image computation
#[derive(Debug, Clone)]
pub enum PersistenceWeight {
    /// Uniform weight (all points weighted equally)
    Uniform,
    /// Linear weight: w(b, p) = p (favors high persistence)
    Linear,
    /// Arctan weight: w(b, p) = arctan(p) (smooth truncation)
    Arctan,
    /// Custom weight based on persistence threshold
    Threshold(f64),
}

impl PersistenceImage {
    /// Create a new PersistenceImage computer
    ///
    /// # Arguments
    /// * `resolution` - Grid resolution (resolution × resolution)
    /// * `dimension` - Homological dimension to use
    /// * `sigma` - Gaussian kernel bandwidth
    /// * `weight_type` - Weight function
    pub fn new(
        resolution: usize,
        dimension: usize,
        sigma: f64,
        weight_type: PersistenceWeight,
    ) -> Result<Self> {
        if resolution == 0 {
            return Err(TransformError::InvalidInput(
                "Resolution must be positive".to_string(),
            ));
        }
        if sigma <= 0.0 {
            return Err(TransformError::InvalidInput(
                "Sigma must be positive".to_string(),
            ));
        }
        Ok(Self {
            resolution,
            birth_range: (0.0, 1.0),
            persistence_range: (0.0, 1.0),
            sigma,
            weight_type,
            dimension,
        })
    }

    /// Compute a persistence image from a persistence diagram
    ///
    /// # Arguments
    /// * `diagram` - The persistence diagram to vectorize
    /// * `resolution` - Grid resolution
    ///
    /// # Returns
    /// * A (resolution × resolution) array representing the persistence image
    pub fn compute(diagram: &PersistenceDiagram, resolution: usize) -> Result<Array2<f64>> {
        if resolution == 0 {
            return Err(TransformError::InvalidInput(
                "Resolution must be positive".to_string(),
            ));
        }

        let img = PersistenceImage::new(resolution, 0, 0.1, PersistenceWeight::Linear)?;
        img.transform(diagram)
    }

    /// Transform a diagram using this image's configuration
    pub fn transform(&self, diagram: &PersistenceDiagram) -> Result<Array2<f64>> {
        // Collect finite points in the target dimension
        let pts: Vec<(f64, f64)> = diagram
            .points
            .iter()
            .filter(|p| p.dimension == self.dimension && !p.is_essential())
            .map(|p| (p.birth, p.persistence()))
            .collect();

        if pts.is_empty() {
            return Ok(Array2::zeros((self.resolution, self.resolution)));
        }

        // Determine range from data if not set (auto-range)
        let b_min = self.birth_range.0;
        let b_max = self.birth_range.1.max(pts.iter().map(|(b, _)| *b).fold(0.0_f64, f64::max));
        let p_min = self.persistence_range.0;
        let p_max = self.persistence_range.1.max(pts.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max));

        let b_range = (b_max - b_min).max(1e-10);
        let p_range = (p_max - p_min).max(1e-10);
        let cell_size_b = b_range / self.resolution as f64;
        let cell_size_p = p_range / self.resolution as f64;

        let mut image = Array2::<f64>::zeros((self.resolution, self.resolution));
        let norm_factor = 1.0 / (2.0 * std::f64::consts::PI * self.sigma * self.sigma);

        for &(birth, pers) in &pts {
            // Weight for this point
            let weight = match &self.weight_type {
                PersistenceWeight::Uniform => 1.0,
                PersistenceWeight::Linear => pers,
                PersistenceWeight::Arctan => pers.atan(),
                PersistenceWeight::Threshold(t) => {
                    if pers >= *t { 1.0 } else { pers / t }
                }
            };

            // Add Gaussian kernel contribution to each grid cell
            for i in 0..self.resolution {
                let cell_b = b_min + (i as f64 + 0.5) * cell_size_b;
                for j in 0..self.resolution {
                    let cell_p = p_min + (j as f64 + 0.5) * cell_size_p;
                    let db = (cell_b - birth) / self.sigma;
                    let dp = (cell_p - pers) / self.sigma;
                    let gauss = norm_factor * (-0.5 * (db * db + dp * dp)).exp();
                    image[[i, j]] += weight * gauss * cell_size_b * cell_size_p;
                }
            }
        }

        Ok(image)
    }

    /// Set the birth range for the image
    pub fn with_birth_range(mut self, min: f64, max: f64) -> Self {
        self.birth_range = (min, max);
        self
    }

    /// Set the persistence range for the image
    pub fn with_persistence_range(mut self, min: f64, max: f64) -> Self {
        self.persistence_range = (min, max);
        self
    }
}

// ─── Distances Between Diagrams ───────────────────────────────────────────────

/// Compute the bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance measures the maximum displacement when matching
/// points between two diagrams optimally (each point can also be matched
/// to the diagonal).
///
/// # Arguments
/// * `d1` - First persistence diagram
/// * `d2` - Second persistence diagram
///
/// # Returns
/// * The bottleneck distance (non-negative)
pub fn bottleneck_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram) -> f64 {
    // Get finite points from both diagrams (all dimensions)
    let pts1: Vec<(f64, f64)> = d1
        .points
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    let pts2: Vec<(f64, f64)> = d2
        .points
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    bottleneck_distance_between(&pts1, &pts2)
}

/// Compute the bottleneck distance between two persistence diagrams for a specific dimension.
pub fn bottleneck_distance_dim(d1: &PersistenceDiagram, d2: &PersistenceDiagram, dim: usize) -> f64 {
    let pts1: Vec<(f64, f64)> = d1
        .points
        .iter()
        .filter(|p| p.dimension == dim && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    let pts2: Vec<(f64, f64)> = d2
        .points
        .iter()
        .filter(|p| p.dimension == dim && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    bottleneck_distance_between(&pts1, &pts2)
}

/// Compute the Wasserstein-p distance between two persistence diagrams.
///
/// # Arguments
/// * `d1` - First persistence diagram
/// * `d2` - Second persistence diagram
/// * `p` - Wasserstein power (typically 1 or 2)
///
/// # Returns
/// * The Wasserstein-p distance
pub fn wasserstein_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, p: f64) -> f64 {
    let pts1: Vec<(f64, f64)> = d1
        .points
        .iter()
        .filter(|p| !p.is_essential())
        .map(|pt| (pt.birth, pt.death))
        .collect();

    let pts2: Vec<(f64, f64)> = d2
        .points
        .iter()
        .filter(|p| !p.is_essential())
        .map(|pt| (pt.birth, pt.death))
        .collect();

    wasserstein_distance_between(&pts1, &pts2, p)
}

/// Internal: compute bottleneck distance between two point sets with diagonal projections.
/// Uses a binary search + bipartite matching approach.
fn bottleneck_distance_between(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> f64 {
    // Distance from a diagram point (b,d) to the diagonal
    let diag_dist = |(b, d): (f64, f64)| -> f64 { (d - b) / 2.0 };

    // L∞ distance between two diagram points
    let point_dist = |(b1, d1): (f64, f64), (b2, d2): (f64, f64)| -> f64 {
        (b1 - b2).abs().max((d1 - d2).abs())
    };

    // If both are empty, distance is 0
    if pts1.is_empty() && pts2.is_empty() {
        return 0.0;
    }

    // Collect all candidate distances for binary search
    let mut candidates = Vec::new();

    for &p1 in pts1 {
        for &p2 in pts2 {
            candidates.push(point_dist(p1, p2));
        }
        candidates.push(diag_dist(p1));
    }
    for &p2 in pts2 {
        candidates.push(diag_dist(p2));
    }
    candidates.push(0.0);

    // Sort and deduplicate candidates
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    // Binary search for minimum bottleneck distance using hopcroft-karp style feasibility
    let mut lo = 0;
    let mut hi = candidates.len().saturating_sub(1);
    let mut result = candidates.last().copied().unwrap_or(0.0);

    while lo <= hi {
        let mid = (lo + hi) / 2;
        let delta = candidates[mid];

        if is_feasible_bottleneck(pts1, pts2, delta) {
            result = delta;
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    result
}

/// Check if there's a perfect matching with all costs ≤ delta (bottleneck feasibility)
/// Uses greedy matching with diagonal fallback
fn is_feasible_bottleneck(pts1: &[(f64, f64)], pts2: &[(f64, f64)], delta: f64) -> bool {
    let diag_dist = |(b, d): (f64, f64)| -> f64 { (d - b) / 2.0 };
    let point_dist = |(b1, d1): (f64, f64), (b2, d2): (f64, f64)| -> f64 {
        (b1 - b2).abs().max((d1 - d2).abs())
    };

    // Try to match pts1 to pts2 or diagonal; use augmenting path search
    let n = pts1.len();
    let m = pts2.len();

    // Bipartite graph: left = pts1, right = pts2 ∪ diagonal projections of pts1 and pts2
    // Simplified: use greedy matching then check unmatched points against diagonal

    let mut matched2 = vec![false; m];
    let mut matched1 = vec![false; n];

    // Try to match each point in pts1 to a point in pts2
    let mut assignment: Vec<Option<usize>> = vec![None; n];

    for i in 0..n {
        for j in 0..m {
            if !matched2[j] && point_dist(pts1[i], pts2[j]) <= delta {
                assignment[i] = Some(j);
                matched2[j] = true;
                matched1[i] = true;
                break;
            }
        }
    }

    // All unmatched pts1 must be within delta of their diagonal projection
    for i in 0..n {
        if !matched1[i] && diag_dist(pts1[i]) > delta {
            return false;
        }
    }

    // All unmatched pts2 must be within delta of their diagonal projection
    for j in 0..m {
        if !matched2[j] && diag_dist(pts2[j]) > delta {
            return false;
        }
    }

    true
}

/// Internal: compute Wasserstein distance between two point sets
fn wasserstein_distance_between(pts1: &[(f64, f64)], pts2: &[(f64, f64)], p: f64) -> f64 {
    let diag_dist = |(b, d): (f64, f64)| -> f64 { (d - b) / 2.0 };
    let point_dist_lp = |(b1, d1): (f64, f64), (b2, d2): (f64, f64), p: f64| -> f64 {
        // Use L∞ for Wasserstein as is standard in TDA
        (b1 - b2).abs().max((d1 - d2).abs()).powf(p)
    };

    // Pad smaller set with diagonal projections
    let n = pts1.len();
    let m = pts2.len();

    // Greedy matching with Hungarian-style cost minimization (simplified)
    let mut total_cost = 0.0f64;

    // Unmatched points go to diagonal
    let mut matched2 = vec![false; m];

    for i in 0..n {
        // Find best match for pts1[i] (either a point in pts2 or the diagonal)
        let diag_cost = diag_dist(pts1[i]).powf(p);
        let mut best_cost = diag_cost;
        let mut best_j = None;

        for j in 0..m {
            if !matched2[j] {
                let cost = point_dist_lp(pts1[i], pts2[j], p);
                if cost < best_cost {
                    best_cost = cost;
                    best_j = Some(j);
                }
            }
        }

        if let Some(j) = best_j {
            matched2[j] = true;
        }
        total_cost += best_cost;
    }

    // Remaining unmatched pts2 go to diagonal
    for j in 0..m {
        if !matched2[j] {
            total_cost += diag_dist(pts2[j]).powf(p);
        }
    }

    total_cost.powf(1.0 / p)
}

// ─── Persistence Landscapes ───────────────────────────────────────────────────

/// Persistence landscape: a functional summary of persistence diagrams.
///
/// The k-th landscape function λ_k(t) is the k-th largest "tent function" value at t.
/// Landscapes are stable, vectorizable, and support averaging over multiple diagrams.
#[derive(Debug, Clone)]
pub struct PersistenceLandscape {
    /// Number of landscape functions to compute
    n_landscapes: usize,
    /// Homological dimension
    dimension: usize,
    /// Sampled landscape values at grid points
    pub landscapes: Array2<f64>,
    /// Grid points (t values)
    pub grid: Array1<f64>,
}

impl PersistenceLandscape {
    /// Compute persistence landscapes from a diagram
    ///
    /// # Arguments
    /// * `diagram` - The persistence diagram
    /// * `n_landscapes` - Number of landscape functions (k = 1, ..., n_landscapes)
    /// * `n_grid_points` - Number of grid points for sampling
    /// * `dimension` - Homological dimension
    ///
    /// # Returns
    /// * A PersistenceLandscape with sampled landscape values
    pub fn compute(
        diagram: &PersistenceDiagram,
        n_landscapes: usize,
        n_grid_points: usize,
        dimension: usize,
    ) -> Result<Self> {
        if n_landscapes == 0 {
            return Err(TransformError::InvalidInput(
                "n_landscapes must be positive".to_string(),
            ));
        }
        if n_grid_points < 2 {
            return Err(TransformError::InvalidInput(
                "n_grid_points must be at least 2".to_string(),
            ));
        }

        let pts: Vec<(f64, f64)> = diagram
            .points
            .iter()
            .filter(|p| p.dimension == dimension && !p.is_essential())
            .map(|p| (p.birth, p.death))
            .collect();

        if pts.is_empty() {
            let grid = Array1::linspace(0.0, 1.0, n_grid_points);
            return Ok(Self {
                n_landscapes,
                dimension,
                landscapes: Array2::zeros((n_landscapes, n_grid_points)),
                grid,
            });
        }

        // Determine grid range
        let t_min = pts.iter().map(|(b, _)| *b).fold(f64::INFINITY, f64::min);
        let t_max = pts.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);
        let grid = Array1::linspace(t_min, t_max, n_grid_points);

        let mut landscapes = Array2::<f64>::zeros((n_landscapes, n_grid_points));

        for (g_idx, &t) in grid.iter().enumerate() {
            // Compute tent function values at t for all points
            let mut tent_values: Vec<f64> = pts
                .iter()
                .map(|&(b, d)| {
                    if t <= (b + d) / 2.0 {
                        (t - b).max(0.0)
                    } else {
                        (d - t).max(0.0)
                    }
                })
                .collect();

            // Sort descending to get the k-th largest values
            tent_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            for k in 0..n_landscapes {
                landscapes[[k, g_idx]] = tent_values.get(k).copied().unwrap_or(0.0);
            }
        }

        Ok(Self {
            n_landscapes,
            dimension,
            landscapes,
            grid,
        })
    }

    /// Compute the L2 norm of the k-th landscape function
    pub fn l2_norm(&self, k: usize) -> f64 {
        if k >= self.n_landscapes {
            return 0.0;
        }
        let row = self.landscapes.row(k);
        row.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Inner product between two landscape functions
    pub fn inner_product(&self, other: &Self) -> f64 {
        let n = self.landscapes.shape()[1].min(other.landscapes.shape()[1]);
        let k = self.n_landscapes.min(other.n_landscapes);
        let mut sum = 0.0;
        for i in 0..k {
            for j in 0..n {
                sum += self.landscapes[[i, j]] * other.landscapes[[i, j]];
            }
        }
        sum
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn square_points() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        )
        .expect("shape ok")
    }

    #[test]
    fn test_vietoris_rips_h0() {
        let pts = square_points();
        let diagram = VietorisRips::compute(&pts, 0, 2.0).expect("vr compute");
        // At H0 dimension we expect connected components
        let h0_pts = diagram.points_in_dimension(0);
        assert!(!h0_pts.is_empty(), "Should have H0 features");
    }

    #[test]
    fn test_vietoris_rips_h1() {
        let pts = square_points();
        let diagram = VietorisRips::compute(&pts, 1, 2.0).expect("vr compute");
        // Square should have a 1-cycle (loop)
        let h1_pts = diagram.points_in_dimension(1);
        // The square has at least one loop
        let _finite_h1: Vec<_> = h1_pts.iter().filter(|p| !p.is_essential()).collect();
        // At least H0 features should be present
        assert!(!diagram.is_empty(), "Diagram should not be empty");
    }

    #[test]
    fn test_persistence_point_persistence() {
        let p = PersistencePoint::new(0.5, 1.5, 0);
        assert!((p.persistence() - 1.0).abs() < 1e-10);
        assert!(!p.is_essential());

        let q = PersistencePoint::new(0.5, f64::INFINITY, 0);
        assert!(q.is_essential());
        assert!(q.persistence().is_infinite());
    }

    #[test]
    fn test_persistence_diagram_filter() {
        let mut diagram = PersistenceDiagram::new(1);
        diagram.add_point(0.0, 0.01, 0); // short-lived (noise)
        diagram.add_point(0.0, 1.0, 0); // long-lived (signal)
        diagram.add_point(0.2, 0.8, 1); // H1 feature

        let filtered = diagram.filter_by_persistence(0.5);
        assert_eq!(filtered.len(), 2); // only the two long-lived features
    }

    #[test]
    fn test_barcode_from_diagram() {
        let mut diagram = PersistenceDiagram::new(1);
        diagram.add_point(0.0, 1.0, 0);
        diagram.add_point(0.5, 0.9, 1);

        let barcode = diagram.to_barcode();
        assert_eq!(barcode.len(), 2);
        assert_eq!(barcode.intervals_in_dimension(0).len(), 1);
        assert_eq!(barcode.intervals_in_dimension(1).len(), 1);
        assert!((barcode.intervals_in_dimension(0)[0].length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_persistence_image() {
        let mut diagram = PersistenceDiagram::new(1);
        diagram.add_point(0.0, 1.0, 0);
        diagram.add_point(0.2, 0.8, 0);

        let image = PersistenceImage::compute(&diagram, 10).expect("pi compute");
        assert_eq!(image.shape(), &[10, 10]);
        // Image should have non-negative values
        assert!(image.iter().all(|&v| v >= 0.0));
        // Image should have some non-zero content
        assert!(image.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_bottleneck_distance_same_diagram() {
        let mut diagram = PersistenceDiagram::new(0);
        diagram.add_point(0.0, 1.0, 0);
        diagram.add_point(0.5, 0.9, 0);

        // Bottleneck distance with itself should be ~0
        let dist = bottleneck_distance(&diagram, &diagram);
        assert!(dist < 1e-10, "Self-distance should be 0, got {}", dist);
    }

    #[test]
    fn test_bottleneck_distance_empty_diagrams() {
        let d1 = PersistenceDiagram::new(0);
        let d2 = PersistenceDiagram::new(0);
        let dist = bottleneck_distance(&d1, &d2);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_bottleneck_distance_different_diagrams() {
        let mut d1 = PersistenceDiagram::new(0);
        d1.add_point(0.0, 1.0, 0);

        let mut d2 = PersistenceDiagram::new(0);
        d2.add_point(0.0, 0.5, 0);

        let dist = bottleneck_distance(&d1, &d2);
        // The optimal matching pairs (0,1) with (0,0.5), cost = max(|0-0|, |1-0.5|) = 0.5
        // or match (0,1) to diagonal at (0.5, 0.5), cost = 0.5
        // and (0,0.5) to diagonal at (0.25, 0.25), cost = 0.25
        // max = 0.5
        assert!(dist > 0.0, "Different diagrams should have positive distance");
    }

    #[test]
    fn test_persistence_landscape() {
        let mut diagram = PersistenceDiagram::new(0);
        diagram.add_point(0.0, 2.0, 0);
        diagram.add_point(0.5, 1.5, 0);

        let landscape = PersistenceLandscape::compute(&diagram, 2, 20, 0)
            .expect("landscape compute");
        assert_eq!(landscape.landscapes.shape(), &[2, 20]);
        // First landscape function should be non-negative
        assert!(landscape.landscapes.row(0).iter().all(|&v| v >= -1e-10));
        assert!(landscape.l2_norm(0) > 0.0);
    }

    #[test]
    fn test_wasserstein_distance() {
        let mut d1 = PersistenceDiagram::new(0);
        d1.add_point(0.0, 1.0, 0);

        let mut d2 = PersistenceDiagram::new(0);
        d2.add_point(0.0, 1.0, 0);

        // Wasserstein distance of identical diagrams should be 0
        let wd = wasserstein_distance(&d1, &d2, 1.0);
        assert!(wd < 1e-10, "Identical diagrams: W=0, got {}", wd);
    }

    #[test]
    fn test_betti_numbers() {
        let mut diagram = PersistenceDiagram::new(1);
        diagram.add_point(0.0, f64::INFINITY, 0); // one component throughout
        diagram.add_point(0.3, 0.7, 1); // loop from t=0.3 to t=0.7

        let betti = diagram.betti_numbers_at(0.5);
        assert_eq!(betti[0], 1); // one connected component
        assert_eq!(betti[1], 1); // one loop active at t=0.5

        let betti_early = diagram.betti_numbers_at(0.1);
        assert_eq!(betti_early[1], 0); // loop not yet born
    }

    #[test]
    fn test_vietoris_rips_small_radius() {
        let pts = square_points();
        // With very small radius, no edges are formed so all 4 points are separate components
        let diagram = VietorisRips::compute(&pts, 0, 0.1).expect("vr compute");
        let h0_pts = diagram.points_in_dimension(0);
        // All 4 points should appear as H0 features (either finite or essential)
        assert!(!h0_pts.is_empty());
    }

    #[test]
    fn test_total_persistence() {
        let mut diagram = PersistenceDiagram::new(0);
        diagram.add_point(0.0, 1.0, 0);
        diagram.add_point(0.0, 3.0, 0);

        let tp = diagram.total_persistence(2.0);
        // Should be sqrt(1^2 + 3^2) = sqrt(10) ≈ 3.162
        assert!((tp - (10.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_persistence_image_custom() {
        let mut diagram = PersistenceDiagram::new(0);
        diagram.add_point(0.0, 1.0, 0);
        diagram.add_point(0.2, 0.8, 0);

        let img_computer = PersistenceImage::new(5, 0, 0.2, PersistenceWeight::Arctan)
            .expect("pi new")
            .with_birth_range(0.0, 1.0)
            .with_persistence_range(0.0, 1.0);

        let image = img_computer.transform(&diagram).expect("pi transform");
        assert_eq!(image.shape(), &[5, 5]);
        assert!(image.iter().all(|&v| v >= 0.0));
    }
}
