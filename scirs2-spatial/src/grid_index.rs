//! Uniform Spatial Grid Index for fixed-radius queries
//!
//! This module provides a grid-based spatial index optimized for
//! fixed-radius nearest neighbor queries and proximity searches.
//! Points are assigned to cells based on their coordinates, enabling
//! O(1) cell lookup and efficient neighborhood enumeration.
//!
//! # Key Features
//!
//! - Uniform spatial grid with configurable cell size
//! - Cell-based point storage with O(1) cell lookup
//! - Fixed-radius nearest neighbor queries (search cell + neighbors)
//! - Dynamic point insertion and deletion
//! - Efficient for uniformly distributed data with known query radius
//!
//! # References
//!
//! * Bentley & Friedman (1979) "Data Structures for Range Searching"
//! * Green (1981) "Grid-based spatial indexing"

use crate::error::{SpatialError, SpatialResult};
use crate::safe_conversions::*;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// CellKey
// ---------------------------------------------------------------------------

/// Integer cell coordinates for a grid cell
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellKey {
    /// Cell indices for each dimension (packed into a fixed-size array)
    coords: [i64; MAX_DIMS],
    /// Active number of dimensions
    dims: usize,
}

/// Maximum supported dimensionality for the grid
const MAX_DIMS: usize = 16;

impl CellKey {
    fn new(coords: &[i64]) -> SpatialResult<Self> {
        if coords.len() > MAX_DIMS {
            return Err(SpatialError::ValueError(format!(
                "Grid index supports at most {} dimensions, got {}",
                MAX_DIMS,
                coords.len()
            )));
        }
        let mut arr = [0i64; MAX_DIMS];
        for (i, &c) in coords.iter().enumerate() {
            arr[i] = c;
        }
        Ok(CellKey {
            coords: arr,
            dims: coords.len(),
        })
    }

    fn dim_coords(&self) -> &[i64] {
        &self.coords[..self.dims]
    }
}

// ---------------------------------------------------------------------------
// StoredPoint
// ---------------------------------------------------------------------------

/// A point stored in the grid with its original data and user ID
#[derive(Clone, Debug)]
struct StoredPoint<T: Float> {
    /// Original index / identifier
    id: usize,
    /// Coordinate values
    coords: Vec<T>,
}

// ---------------------------------------------------------------------------
// GridIndex
// ---------------------------------------------------------------------------

/// Uniform spatial grid index for efficient fixed-radius queries.
///
/// The grid divides space into uniform cells of a configurable size.
/// Points are hashed into cells, and queries search only the relevant
/// cells (the target cell and its neighbors), making the query cost
/// proportional to the local point density rather than the total dataset
/// size.
///
/// # Type Parameters
///
/// * `T` - Floating-point type (f32 or f64)
///
/// # Example
///
/// ```rust
/// use scirs2_spatial::grid_index::GridIndex;
///
/// let mut grid = GridIndex::<f64>::new(1.0, 2).unwrap();
/// grid.insert(0, &[0.0, 0.0]).unwrap();
/// grid.insert(1, &[0.5, 0.5]).unwrap();
/// grid.insert(2, &[3.0, 3.0]).unwrap();
///
/// let (ids, dists) = grid.query_radius(&[0.0, 0.0], 1.0).unwrap();
/// assert_eq!(ids.len(), 2); // points 0 and 1
/// ```
#[derive(Clone, Debug)]
pub struct GridIndex<T: Float> {
    /// Cell size (same for all dimensions)
    cell_size: T,
    /// Inverse cell size for fast division
    inv_cell_size: T,
    /// Dimensionality of points
    n_dims: usize,
    /// Map from cell key to the list of points in that cell
    cells: HashMap<CellKey, Vec<StoredPoint<T>>>,
    /// Total number of points currently stored
    count: usize,
    /// Phantom
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> GridIndex<T> {
    /// Create a new empty grid index.
    ///
    /// # Arguments
    ///
    /// * `cell_size` - The side length of each grid cell.  For optimal
    ///   performance with radius-r queries, set `cell_size = r`.
    /// * `n_dims` - Number of spatial dimensions
    pub fn new(cell_size: f64, n_dims: usize) -> SpatialResult<Self> {
        if cell_size <= 0.0 {
            return Err(SpatialError::ValueError(
                "Cell size must be positive".to_string(),
            ));
        }
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(SpatialError::ValueError(format!(
                "Dimensions must be in [1, {}], got {}",
                MAX_DIMS, n_dims
            )));
        }
        let cs: T = safe_from(cell_size, "grid cell_size")?;
        let ics: T = safe_from(1.0 / cell_size, "grid inv_cell_size")?;
        Ok(GridIndex {
            cell_size: cs,
            inv_cell_size: ics,
            n_dims,
            cells: HashMap::new(),
            count: 0,
            _phantom: PhantomData,
        })
    }

    /// Return the number of points in the grid.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return `true` if the grid contains no points.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Return the number of non-empty cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Return the dimensionality.
    pub fn dims(&self) -> usize {
        self.n_dims
    }

    /// Return the cell size.
    pub fn cell_size(&self) -> T {
        self.cell_size
    }

    // ------------------------------------------------------------------
    // Insertion / deletion
    // ------------------------------------------------------------------

    /// Insert a point into the grid.
    ///
    /// # Arguments
    ///
    /// * `id` - A user-provided identifier for the point
    /// * `coords` - The coordinates of the point (length must equal `n_dims`)
    pub fn insert(&mut self, id: usize, coords: &[T]) -> SpatialResult<()> {
        if coords.len() != self.n_dims {
            return Err(SpatialError::DimensionError(format!(
                "Expected {} dims, got {}",
                self.n_dims,
                coords.len()
            )));
        }
        let key = self.cell_key(coords)?;
        let sp = StoredPoint {
            id,
            coords: coords.to_vec(),
        };
        self.cells.entry(key).or_default().push(sp);
        self.count += 1;
        Ok(())
    }

    /// Insert many points at once from a slice of (id, coords) pairs.
    pub fn insert_batch(&mut self, points: &[(usize, Vec<T>)]) -> SpatialResult<()> {
        for (id, coords) in points {
            self.insert(*id, coords)?;
        }
        Ok(())
    }

    /// Remove a point by its `id`. Returns `true` if a point was removed.
    ///
    /// If there are multiple points with the same `id`, only the first
    /// occurrence is removed.
    pub fn remove(&mut self, id: usize) -> bool {
        for cell in self.cells.values_mut() {
            if let Some(pos) = cell.iter().position(|sp| sp.id == id) {
                cell.swap_remove(pos);
                self.count -= 1;
                return true;
            }
        }
        false
    }

    /// Remove all points, keeping the grid structure.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.count = 0;
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Find all points within `radius` of `query`.
    ///
    /// Returns `(ids, distances)` sorted by ascending distance.
    pub fn query_radius(&self, query: &[T], radius: f64) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if query.len() != self.n_dims {
            return Err(SpatialError::DimensionError(format!(
                "Expected {} dims, got {}",
                self.n_dims,
                query.len()
            )));
        }
        if radius < 0.0 {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        let r: T = safe_from(radius, "query radius")?;
        let r_sq = r * r;

        // How many cells we need to check in each dimension
        let cell_radius = (radius / self.to_f64(self.cell_size)).ceil() as i64;

        let center_key = self.cell_key(query)?;
        let mut results: Vec<(usize, T)> = Vec::new();

        // Iterate over all neighbor cells
        self.for_each_neighbor_cell(&center_key, cell_radius, |cell_points| {
            for sp in cell_points {
                let d_sq = self.squared_distance(query, &sp.coords);
                if d_sq <= r_sq {
                    results.push((sp.id, d_sq.sqrt()));
                }
            }
        });

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (ids, dists): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        Ok((ids, dists))
    }

    /// Find the `k` nearest neighbors of `query` within the grid.
    ///
    /// This performs an expanding cell search: it first checks the local
    /// cell and its immediate neighbors, then expands if not enough points
    /// are found. Returns `(ids, distances)` sorted by distance.
    pub fn query_knn(&self, query: &[T], k: usize) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if query.len() != self.n_dims {
            return Err(SpatialError::DimensionError(format!(
                "Expected {} dims, got {}",
                self.n_dims,
                query.len()
            )));
        }
        if k == 0 {
            return Ok((vec![], vec![]));
        }

        let k = k.min(self.count);
        let center_key = self.cell_key(query)?;

        // Expanding search: start from cell_radius=1 and grow
        let mut cell_radius: i64 = 1;
        let max_radius: i64 = 64; // prevent infinite loops

        loop {
            let mut candidates: Vec<(usize, T)> = Vec::new();

            self.for_each_neighbor_cell(&center_key, cell_radius, |cell_points| {
                for sp in cell_points {
                    let d = self.euclidean_distance(query, &sp.coords);
                    candidates.push((sp.id, d));
                }
            });

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if candidates.len() >= k || cell_radius >= max_radius {
                candidates.truncate(k);
                let (ids, dists): (Vec<_>, Vec<_>) = candidates.into_iter().unzip();
                return Ok((ids, dists));
            }

            cell_radius += 1;
        }
    }

    /// Check whether any point exists within `radius` of `query`.
    pub fn has_neighbor(&self, query: &[T], radius: f64) -> SpatialResult<bool> {
        if query.len() != self.n_dims {
            return Err(SpatialError::DimensionError(format!(
                "Expected {} dims, got {}",
                self.n_dims,
                query.len()
            )));
        }
        let r: T = safe_from(radius, "has_neighbor radius")?;
        let r_sq = r * r;
        let cell_radius = (radius / self.to_f64(self.cell_size)).ceil() as i64;
        let center_key = self.cell_key(query)?;

        let mut found = false;
        self.for_each_neighbor_cell(&center_key, cell_radius, |cell_points| {
            if found {
                return;
            }
            for sp in cell_points {
                let d_sq = self.squared_distance(query, &sp.coords);
                if d_sq <= r_sq {
                    found = true;
                    return;
                }
            }
        });

        Ok(found)
    }

    /// Return all point ids stored in the grid.
    pub fn all_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.count);
        for cell in self.cells.values() {
            for sp in cell {
                ids.push(sp.id);
            }
        }
        ids
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the cell key for given coordinates.
    fn cell_key(&self, coords: &[T]) -> SpatialResult<CellKey> {
        let mut ints = Vec::with_capacity(self.n_dims);
        for &c in coords.iter().take(self.n_dims) {
            let idx = (c * self.inv_cell_size).floor();
            let i = self.to_i64(idx);
            ints.push(i);
        }
        CellKey::new(&ints)
    }

    /// Iterate over all cells within `cell_radius` of `center_key`,
    /// calling `f` with the points in each non-empty cell.
    fn for_each_neighbor_cell<F>(&self, center_key: &CellKey, cell_radius: i64, mut f: F)
    where
        F: FnMut(&[StoredPoint<T>]),
    {
        let dims = center_key.dims;
        let center = center_key.dim_coords();

        // Generate all neighbor offsets (up to cell_radius in each dim)
        let mut offsets = vec![vec![0i64; dims]];
        for d in 0..dims {
            let mut new_offsets = Vec::new();
            for existing in &offsets {
                for delta in -cell_radius..=cell_radius {
                    let mut combo = existing.clone();
                    combo[d] = center[d] + delta;
                    new_offsets.push(combo);
                }
            }
            offsets = new_offsets;
        }

        for offset in &offsets {
            if let Ok(key) = CellKey::new(offset) {
                if let Some(cell_points) = self.cells.get(&key) {
                    if !cell_points.is_empty() {
                        f(cell_points);
                    }
                }
            }
        }
    }

    fn squared_distance(&self, a: &[T], b: &[T]) -> T {
        let mut sum = T::zero();
        for i in 0..self.n_dims {
            let d = a[i] - b[i];
            sum = sum + d * d;
        }
        sum
    }

    fn euclidean_distance(&self, a: &[T], b: &[T]) -> T {
        self.squared_distance(a, b).sqrt()
    }

    fn to_f64(&self, val: T) -> f64 {
        val.to_f64().unwrap_or(0.0)
    }

    fn to_i64(&self, val: T) -> i64 {
        val.to_f64().unwrap_or(0.0) as i64
    }
}

// ---------------------------------------------------------------------------
// Convenience: build from a 2D array
// ---------------------------------------------------------------------------

impl<T: Float + 'static> GridIndex<T> {
    /// Build a grid index from an (n x d) array of points.
    ///
    /// Point ids will be 0..n.
    pub fn from_array(
        data: &scirs2_core::ndarray::ArrayView2<T>,
        cell_size: f64,
    ) -> SpatialResult<Self> {
        let n = data.nrows();
        let d = data.ncols();
        let mut grid = GridIndex::new(cell_size, d)?;
        for i in 0..n {
            let row = data.row(i).to_vec();
            grid.insert(i, &row)?;
        }
        Ok(grid)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_create_grid() {
        let grid = GridIndex::<f64>::new(1.0, 2);
        assert!(grid.is_ok());
        let grid = grid.expect("should create");
        assert_eq!(grid.len(), 0);
        assert!(grid.is_empty());
        assert_eq!(grid.dims(), 2);
    }

    #[test]
    fn test_invalid_cell_size() {
        let result = GridIndex::<f64>::new(0.0, 2);
        assert!(result.is_err());
        let result = GridIndex::<f64>::new(-1.0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_dims() {
        let result = GridIndex::<f64>::new(1.0, 0);
        assert!(result.is_err());
        let result = GridIndex::<f64>::new(1.0, 17);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_and_count() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.5, 0.5]).expect("insert");
        grid.insert(1, &[1.5, 0.5]).expect("insert");
        grid.insert(2, &[0.5, 1.5]).expect("insert");
        assert_eq!(grid.len(), 3);
        assert!(!grid.is_empty());
    }

    #[test]
    fn test_insert_wrong_dims() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        let result = grid.insert(0, &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");
        grid.insert(1, &[1.0, 1.0]).expect("insert");
        assert_eq!(grid.len(), 2);

        assert!(grid.remove(0));
        assert_eq!(grid.len(), 1);

        // Removing again should fail
        assert!(!grid.remove(0));
        assert_eq!(grid.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");
        grid.insert(1, &[1.0, 1.0]).expect("insert");
        grid.clear();
        assert_eq!(grid.len(), 0);
        assert!(grid.is_empty());
    }

    #[test]
    fn test_query_radius_basic() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");
        grid.insert(1, &[0.5, 0.5]).expect("insert");
        grid.insert(2, &[3.0, 3.0]).expect("insert");

        let (ids, dists) = grid.query_radius(&[0.0, 0.0], 1.0).expect("query");
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));

        // Check distances are sorted
        for i in 1..dists.len() {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_query_radius_empty() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");

        let (ids, _) = grid.query_radius(&[10.0, 10.0], 0.5).expect("query");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_query_radius_negative() {
        let grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        let result = grid.query_radius(&[0.0, 0.0], -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_knn_basic() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");
        grid.insert(1, &[1.0, 0.0]).expect("insert");
        grid.insert(2, &[0.0, 1.0]).expect("insert");
        grid.insert(3, &[10.0, 10.0]).expect("insert");

        let (ids, dists) = grid.query_knn(&[0.1, 0.1], 2).expect("knn");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 0); // closest
                               // Distances should be sorted
        assert!(dists[0] <= dists[1]);
    }

    #[test]
    fn test_knn_k_zero() {
        let grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        let (ids, dists) = grid.query_knn(&[0.0, 0.0], 0).expect("knn");
        assert!(ids.is_empty());
        assert!(dists.is_empty());
    }

    #[test]
    fn test_has_neighbor() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0, 0.0]).expect("insert");

        assert!(grid.has_neighbor(&[0.5, 0.0], 1.0).expect("check"));
        assert!(!grid.has_neighbor(&[5.0, 5.0], 1.0).expect("check"));
    }

    #[test]
    fn test_from_array() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let grid = GridIndex::<f64>::from_array(&pts.view(), 1.0).expect("build");
        assert_eq!(grid.len(), 4);
    }

    #[test]
    fn test_all_ids() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(10, &[0.0, 0.0]).expect("insert");
        grid.insert(20, &[1.0, 1.0]).expect("insert");
        grid.insert(30, &[2.0, 2.0]).expect("insert");

        let mut ids = grid.all_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_cell_count() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        // Points in same cell
        grid.insert(0, &[0.1, 0.1]).expect("insert");
        grid.insert(1, &[0.2, 0.2]).expect("insert");
        // Point in different cell
        grid.insert(2, &[1.5, 1.5]).expect("insert");

        assert_eq!(grid.cell_count(), 2);
    }

    #[test]
    fn test_3d_grid() {
        let mut grid = GridIndex::<f64>::new(1.0, 3).expect("create");
        grid.insert(0, &[0.0, 0.0, 0.0]).expect("insert");
        grid.insert(1, &[0.5, 0.5, 0.5]).expect("insert");
        grid.insert(2, &[5.0, 5.0, 5.0]).expect("insert");

        let (ids, _) = grid.query_radius(&[0.0, 0.0, 0.0], 1.0).expect("query");
        // sqrt(0.75) ~ 0.866 < 1.0
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_f32_grid() {
        let mut grid = GridIndex::<f32>::new(1.0, 2).expect("create");
        grid.insert(0, &[0.0f32, 0.0]).expect("insert");
        grid.insert(1, &[0.5f32, 0.5]).expect("insert");

        let (ids, dists) = grid.query_radius(&[0.0f32, 0.0], 1.0).expect("query");
        assert_eq!(ids.len(), 2);
        assert_relative_eq!(dists[0], 0.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_coords() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        grid.insert(0, &[-1.0, -1.0]).expect("insert");
        grid.insert(1, &[-0.5, -0.5]).expect("insert");
        grid.insert(2, &[0.5, 0.5]).expect("insert");

        let (ids, _) = grid.query_radius(&[-0.8, -0.8], 0.5).expect("query");
        assert!(ids.contains(&0) || ids.contains(&1));
    }

    #[test]
    fn test_insert_batch() {
        let mut grid = GridIndex::<f64>::new(1.0, 2).expect("create");
        let points = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 1.0]),
            (2, vec![2.0, 2.0]),
        ];
        grid.insert_batch(&points).expect("batch insert");
        assert_eq!(grid.len(), 3);
    }

    #[test]
    fn test_large_dataset() {
        let n = 500;
        let mut grid = GridIndex::<f64>::new(0.1, 2).expect("create");
        for i in 0..n {
            let x = (i as f64) * 0.01;
            let y = (i as f64) * 0.02;
            grid.insert(i, &[x, y]).expect("insert");
        }
        assert_eq!(grid.len(), n);

        // Query near origin
        let (ids, _) = grid.query_radius(&[0.0, 0.0], 0.5).expect("query");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_knn_expanding_search() {
        let mut grid = GridIndex::<f64>::new(0.1, 2).expect("create");
        // Points far from origin
        grid.insert(0, &[5.0, 5.0]).expect("insert");
        grid.insert(1, &[5.1, 5.1]).expect("insert");

        // Query at origin - should still find them via expanding search
        let (ids, _) = grid.query_knn(&[0.0, 0.0], 2).expect("knn");
        assert_eq!(ids.len(), 2);
    }
}
