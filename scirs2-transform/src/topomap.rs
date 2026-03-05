//! Topological Feature Maps: Self-Organizing Map (SOM) based dimensionality reduction
//!
//! This module implements Self-Organizing Maps (SOMs) for unsupervised learning
//! and topological analysis of high-dimensional data. A SOM is a type of
//! artificial neural network that is trained using competitive learning to
//! produce a low-dimensional (typically 2D) discretized representation
//! of the input space.
//!
//! ## Key Concepts
//!
//! - **Prototype vectors**: Each node in the 2D grid stores a reference vector
//!   in the input space.
//! - **Best Matching Unit (BMU)**: The grid node whose prototype is closest
//!   to a given input sample.
//! - **Neighborhood function**: Gaussian kernel centred on the BMU that
//!   controls how strongly each node is updated.
//! - **U-matrix**: The unified distance matrix — visualizes cluster boundaries
//!   by showing distances between adjacent prototype vectors.
//!
//! ## References
//!
//! - Kohonen, T. (2001). Self-Organizing Maps (3rd ed.). Springer.
//! - Ultsch, A., & Siemon, H. P. (1990). Kohonen's Self-Organizing Feature
//!   Maps for Exploratory Data Analysis.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array2, ArrayView1};

// ─── Core Data Structure ──────────────────────────────────────────────────────

/// A trained Self-Organizing Map (topological feature map).
///
/// The map is a 2D grid of `rows × cols` prototype vectors, each of length
/// `n_features`.  After training, nearby nodes in the grid correspond to
/// similar regions of the input space, preserving the topology of the data
/// manifold.
#[derive(Debug, Clone)]
pub struct TopologicalMap {
    /// Number of rows in the grid
    pub rows: usize,
    /// Number of columns in the grid
    pub cols: usize,
    /// Number of input features
    pub n_features: usize,
    /// Prototype vectors stored as a 3-D array of shape `[rows, cols, n_features]`.
    /// Indexing convention: `codebook[[r, c * n_features + f]]` — but we store it
    /// flattened as `Array2<f64>` of shape `(rows * cols, n_features)` for easy
    /// arithmetic with ndarray.
    pub codebook: Array2<f64>,
    /// Quantization error recorded during training (one value per epoch)
    pub training_qe: Vec<f64>,
}

impl TopologicalMap {
    /// Create a new, **untrained** map with random prototype vectors drawn
    /// uniformly from `[0, 1)`.
    ///
    /// Prefer calling [`fit`] directly, which handles initialisation internally.
    pub fn new(rows: usize, cols: usize, n_features: usize) -> Result<Self> {
        if rows == 0 || cols == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput(
                "rows, cols and n_features must all be > 0".to_string(),
            ));
        }
        let n_nodes = rows * cols;
        let codebook = Array2::<f64>::zeros((n_nodes, n_features));
        Ok(Self {
            rows,
            cols,
            n_features,
            codebook,
            training_qe: Vec::new(),
        })
    }

    /// Return the prototype vector for grid node `(r, c)`.
    pub fn prototype(&self, r: usize, c: usize) -> ArrayView1<'_, f64> {
        let idx = r * self.cols + c;
        self.codebook.row(idx)
    }

    /// Number of nodes
    pub fn n_nodes(&self) -> usize {
        self.rows * self.cols
    }
}

// ─── Training ─────────────────────────────────────────────────────────────────

/// Train a Self-Organizing Map on `data`.
///
/// # Parameters
///
/// * `data`            – Input matrix of shape `(n_samples, n_features)`.
/// * `grid_size`       – `(rows, cols)` of the output map.
/// * `n_iter`          – Number of training epochs (full passes through data).
/// * `initial_lr`      – Starting learning rate η₀ (decays linearly to 0).
/// * `initial_radius`  – Starting neighbourhood radius σ₀ (decays
///                        exponentially with a half-life of `n_iter / 2`).
///
/// # Returns
///
/// A fitted [`TopologicalMap`].
pub fn fit(
    data: &Array2<f64>,
    grid_size: (usize, usize),
    n_iter: usize,
    initial_lr: f64,
    initial_radius: f64,
) -> Result<TopologicalMap> {
    let (n_samples, n_features) = (data.nrows(), data.ncols());
    if n_samples == 0 || n_features == 0 {
        return Err(TransformError::InvalidInput(
            "data must have at least one sample and one feature".to_string(),
        ));
    }
    if n_iter == 0 {
        return Err(TransformError::InvalidInput(
            "n_iter must be > 0".to_string(),
        ));
    }
    if initial_lr <= 0.0 || initial_lr > 1.0 {
        return Err(TransformError::InvalidInput(
            "initial_lr must be in (0, 1]".to_string(),
        ));
    }
    if initial_radius <= 0.0 {
        return Err(TransformError::InvalidInput(
            "initial_radius must be > 0".to_string(),
        ));
    }

    let (rows, cols) = grid_size;
    let mut map = TopologicalMap::new(rows, cols, n_features)?;

    // ── Initialise prototype vectors with data statistics ───────────────────
    // Use the per-feature min/max range from the training data so that
    // prototypes are seeded within the data cloud (better than pure random).
    let mut feat_min = vec![f64::INFINITY; n_features];
    let mut feat_max = vec![f64::NEG_INFINITY; n_features];
    for i in 0..n_samples {
        for f in 0..n_features {
            let v = data[[i, f]];
            if v < feat_min[f] {
                feat_min[f] = v;
            }
            if v > feat_max[f] {
                feat_max[f] = v;
            }
        }
    }

    // Simple linear initialisation along the first feature's range
    let n_nodes = rows * cols;
    for node in 0..n_nodes {
        let t = if n_nodes > 1 {
            node as f64 / (n_nodes - 1) as f64
        } else {
            0.5
        };
        for f in 0..n_features {
            let range = feat_max[f] - feat_min[f];
            // Use a deterministic spread rather than random to ensure
            // reproducibility without requiring a random seed parameter.
            map.codebook[[node, f]] = feat_min[f] + t * range;
        }
    }

    // ── Training loop ────────────────────────────────────────────────────────
    let time_constant = n_iter as f64 / initial_radius.ln().max(1.0);

    for epoch in 0..n_iter {
        let progress = epoch as f64 / n_iter as f64;
        let lr = initial_lr * (1.0 - progress); // linear decay
        let sigma = initial_radius * (-(epoch as f64) / time_constant).exp();

        let mut epoch_sq_err = 0.0f64;

        for sample_idx in 0..n_samples {
            let sample = data.row(sample_idx);

            // Find BMU
            let (bmu_r, bmu_c) = find_bmu_internal(&map, sample);

            // Update all prototypes according to neighbourhood function
            for r in 0..rows {
                for c in 0..cols {
                    let h = gaussian_neighbourhood((bmu_r, bmu_c), (r, c), sigma);
                    if h < 1e-10 {
                        continue; // skip negligible updates
                    }
                    let node = r * cols + c;
                    let update_factor = lr * h;
                    for f in 0..n_features {
                        let diff = sample[f] - map.codebook[[node, f]];
                        map.codebook[[node, f]] += update_factor * diff;
                    }
                }
            }

            // Accumulate squared distance to BMU for quantization error
            let bmu_node = bmu_r * cols + bmu_c;
            let sq: f64 = (0..n_features)
                .map(|f| {
                    let d = sample[f] - map.codebook[[bmu_node, f]];
                    d * d
                })
                .sum();
            epoch_sq_err += sq;
        }

        let qe = (epoch_sq_err / n_samples as f64).sqrt();
        map.training_qe.push(qe);
    }

    Ok(map)
}

// ─── Best Matching Unit ───────────────────────────────────────────────────────

/// Find the Best Matching Unit (BMU) coordinates `(row, col)` for a sample.
///
/// The BMU is the grid node whose prototype vector is closest to `sample`
/// in Euclidean distance.
pub fn find_bmu(map: &TopologicalMap, sample: ArrayView1<'_, f64>) -> Result<(usize, usize)> {
    if sample.len() != map.n_features {
        return Err(TransformError::InvalidInput(format!(
            "sample has {} features but map expects {}",
            sample.len(),
            map.n_features
        )));
    }
    Ok(find_bmu_internal(map, sample))
}

/// Internal BMU search (no bounds-checking, caller must ensure dimensionality).
fn find_bmu_internal(map: &TopologicalMap, sample: ArrayView1<'_, f64>) -> (usize, usize) {
    let mut best_dist = f64::INFINITY;
    let mut best_node = 0usize;

    for node in 0..map.n_nodes() {
        let sq: f64 = (0..map.n_features)
            .map(|f| {
                let d = sample[f] - map.codebook[[node, f]];
                d * d
            })
            .sum();
        if sq < best_dist {
            best_dist = sq;
            best_node = node;
        }
    }

    let r = best_node / map.cols;
    let c = best_node % map.cols;
    (r, c)
}

/// Find the second-best matching unit for a sample (used by topographic error).
///
/// Returns `(row, col)` of the second-closest prototype vector.
fn find_second_bmu(map: &TopologicalMap, sample: ArrayView1<'_, f64>) -> (usize, usize) {
    let mut best = (f64::INFINITY, 0usize);
    let mut second = (f64::INFINITY, 0usize);

    for node in 0..map.n_nodes() {
        let sq: f64 = (0..map.n_features)
            .map(|f| {
                let d = sample[f] - map.codebook[[node, f]];
                d * d
            })
            .sum();
        if sq < best.0 {
            second = best;
            best = (sq, node);
        } else if sq < second.0 {
            second = (sq, node);
        }
    }

    let r = second.1 / map.cols;
    let c = second.1 % map.cols;
    (r, c)
}

// ─── Neighbourhood Function ───────────────────────────────────────────────────

/// Gaussian neighbourhood function `h(winner, node, σ)`.
///
/// Returns a value in `[0, 1]` representing how strongly the node at `node_pos`
/// should be updated when the BMU is at `winner_pos`, given a neighbourhood
/// radius (standard deviation) of `sigma`.
///
/// `h = exp( -‖winner − node‖² / (2 σ²) )`
pub fn gaussian_neighbourhood(
    winner_pos: (usize, usize),
    node_pos: (usize, usize),
    sigma: f64,
) -> f64 {
    if sigma < 1e-12 {
        // Degenerate case: only the exact winner is updated
        return if winner_pos == node_pos { 1.0 } else { 0.0 };
    }
    let dr = winner_pos.0 as f64 - node_pos.0 as f64;
    let dc = winner_pos.1 as f64 - node_pos.1 as f64;
    let dist_sq = dr * dr + dc * dc;
    let two_sigma_sq = 2.0 * sigma * sigma;
    (-dist_sq / two_sigma_sq).exp()
}

// ─── Projection ───────────────────────────────────────────────────────────────

/// Project each data point to its BMU coordinates.
///
/// # Returns
///
/// Array of shape `(n_samples, 2)` where each row is `[row_index, col_index]`
/// of the BMU for that sample (stored as `f64` for API compatibility with the
/// rest of the transform crate's dimensionality-reduction functions).
pub fn project(map: &TopologicalMap, data: &Array2<f64>) -> Result<Array2<usize>> {
    if data.ncols() != map.n_features {
        return Err(TransformError::InvalidInput(format!(
            "data has {} features but map expects {}",
            data.ncols(),
            map.n_features
        )));
    }
    let n = data.nrows();
    let mut out = Array2::<usize>::zeros((n, 2));
    for i in 0..n {
        let (r, c) = find_bmu_internal(map, data.row(i));
        out[[i, 0]] = r;
        out[[i, 1]] = c;
    }
    Ok(out)
}

// ─── U-matrix ─────────────────────────────────────────────────────────────────

/// Compute the Unified Distance Matrix (U-matrix).
///
/// The U-matrix visualises the distance structure of the SOM by showing
/// the average Euclidean distance between adjacent prototype vectors.
/// High U-matrix values indicate cluster boundaries; low values indicate
/// regions within clusters.
///
/// # Returns
///
/// Array of shape `(rows, cols)` where each entry is the mean distance
/// from that node's prototype to all its direct (4-connected) neighbours.
pub fn unified_distance_matrix(map: &TopologicalMap) -> Array2<f64> {
    let rows = map.rows;
    let cols = map.cols;
    let mut umat = Array2::<f64>::zeros((rows, cols));

    for r in 0..rows {
        for c in 0..cols {
            let node = r * cols + c;
            let proto = map.codebook.row(node);
            let mut total_dist = 0.0f64;
            let mut n_neighbours = 0usize;

            // 4-connected neighbourhood: up, down, left, right
            let neighbour_offsets: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dr, dc) in neighbour_offsets {
                let nr = r as i64 + dr;
                let nc = c as i64 + dc;
                if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
                    let nb_node = nr as usize * cols + nc as usize;
                    let nb_proto = map.codebook.row(nb_node);
                    let dist: f64 = proto
                        .iter()
                        .zip(nb_proto.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    total_dist += dist;
                    n_neighbours += 1;
                }
            }

            umat[[r, c]] = if n_neighbours > 0 {
                total_dist / n_neighbours as f64
            } else {
                0.0
            };
        }
    }

    umat
}

// ─── Quality Metrics ──────────────────────────────────────────────────────────

/// Compute the quantization error of the map on a dataset.
///
/// Quantization error is the mean Euclidean distance from each sample to
/// its BMU prototype vector.  Lower values indicate a better-fitting map.
///
/// QE = (1/N) Σᵢ ‖xᵢ − w_BMU(xᵢ)‖
pub fn quantization_error(map: &TopologicalMap, data: &Array2<f64>) -> Result<f64> {
    if data.ncols() != map.n_features {
        return Err(TransformError::InvalidInput(format!(
            "data has {} features but map expects {}",
            data.ncols(),
            map.n_features
        )));
    }
    let n = data.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let mut total = 0.0f64;
    for i in 0..n {
        let sample = data.row(i);
        let (br, bc) = find_bmu_internal(map, sample);
        let bmu_node = br * map.cols + bc;
        let dist_sq: f64 = (0..map.n_features)
            .map(|f| {
                let d = sample[f] - map.codebook[[bmu_node, f]];
                d * d
            })
            .sum();
        total += dist_sq.sqrt();
    }
    Ok(total / n as f64)
}

/// Compute the topographic error of the map on a dataset.
///
/// The topographic error measures how well the SOM preserves local topology.
/// For each sample, it checks whether its BMU and second-BMU are adjacent
/// on the 2D grid.  The topographic error is the fraction of samples for
/// which they are **not** adjacent.
///
/// TE = (1/N) Σᵢ 𝟙[BMU(xᵢ) and 2nd-BMU(xᵢ) are not adjacent]
///
/// Values close to 0 indicate good topographic preservation.
pub fn topographic_error(map: &TopologicalMap, data: &Array2<f64>) -> Result<f64> {
    if data.ncols() != map.n_features {
        return Err(TransformError::InvalidInput(format!(
            "data has {} features but map expects {}",
            data.ncols(),
            map.n_features
        )));
    }
    let n = data.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let mut errors = 0usize;
    for i in 0..n {
        let sample = data.row(i);
        let (br1, bc1) = find_bmu_internal(map, sample);
        let (br2, bc2) = find_second_bmu(map, sample);

        // Check adjacency: neighbours differ by exactly 1 in one dimension
        let row_diff = (br1 as i64 - br2 as i64).unsigned_abs() as usize;
        let col_diff = (bc1 as i64 - bc2 as i64).unsigned_abs() as usize;
        let adjacent = (row_diff == 1 && col_diff == 0) || (row_diff == 0 && col_diff == 1);
        if !adjacent {
            errors += 1;
        }
    }
    Ok(errors as f64 / n as f64)
}

// ─── Convenience Wrapper ──────────────────────────────────────────────────────

/// Train a SOM using default hyperparameters derived from the data and grid size.
///
/// This is a convenience wrapper around [`fit`] that computes sensible defaults:
/// - `n_iter = 100`
/// - `initial_lr = 0.5`
/// - `initial_radius = max(rows, cols) / 2` (covers half the grid)
pub fn fit_default(data: &Array2<f64>, grid_size: (usize, usize)) -> Result<TopologicalMap> {
    let (rows, cols) = grid_size;
    let initial_radius = (rows.max(cols) as f64) / 2.0;
    fit(data, grid_size, 100, 0.5, initial_radius.max(1.0))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_data() -> Array2<f64> {
        // Four clusters in 2D
        let mut rows = Vec::new();
        for &(cx, cy) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            for j in 0..5 {
                rows.push(vec![cx + j as f64 * 0.05, cy + j as f64 * 0.05]);
            }
        }
        let flat: Vec<f64> = rows.iter().flatten().cloned().collect();
        Array2::from_shape_vec((rows.len(), 2), flat).expect("shape ok")
    }

    #[test]
    fn test_fit_basic() {
        let data = make_data();
        let map = fit(&data, (4, 4), 10, 0.5, 2.0).expect("fit should succeed");
        assert_eq!(map.rows, 4);
        assert_eq!(map.cols, 4);
        assert_eq!(map.n_features, 2);
        assert_eq!(map.training_qe.len(), 10);
    }

    #[test]
    fn test_find_bmu() {
        let data = make_data();
        let map = fit(&data, (4, 4), 20, 0.5, 2.0).expect("fit ok");
        let sample = array![0.0, 0.0];
        let (r, c) = find_bmu(&map, sample.view()).expect("bmu ok");
        assert!(r < 4 && c < 4);
    }

    #[test]
    fn test_project_shape() {
        let data = make_data();
        let map = fit(&data, (4, 4), 20, 0.5, 2.0).expect("fit ok");
        let proj = project(&map, &data).expect("project ok");
        assert_eq!(proj.nrows(), data.nrows());
        assert_eq!(proj.ncols(), 2);
    }

    #[test]
    fn test_umatrix_shape() {
        let data = make_data();
        let map = fit(&data, (4, 4), 20, 0.5, 2.0).expect("fit ok");
        let umat = unified_distance_matrix(&map);
        assert_eq!(umat.shape(), &[4, 4]);
        // All values should be non-negative
        assert!(umat.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_quantization_error_nonneg() {
        let data = make_data();
        let map = fit(&data, (4, 4), 20, 0.5, 2.0).expect("fit ok");
        let qe = quantization_error(&map, &data).expect("qe ok");
        assert!(qe >= 0.0);
    }

    #[test]
    fn test_topographic_error_range() {
        let data = make_data();
        let map = fit(&data, (4, 4), 20, 0.5, 2.0).expect("fit ok");
        let te = topographic_error(&map, &data).expect("te ok");
        assert!((0.0..=1.0).contains(&te));
    }

    #[test]
    fn test_gaussian_neighbourhood() {
        // Winner and node coincide → h = 1
        assert!((gaussian_neighbourhood((2, 2), (2, 2), 1.0) - 1.0).abs() < 1e-10);
        // Far away → h < 1
        assert!(gaussian_neighbourhood((0, 0), (10, 10), 1.0) < 1e-6);
        // Degenerate sigma → only exact winner
        assert_eq!(gaussian_neighbourhood((1, 1), (1, 1), 0.0), 1.0);
        assert_eq!(gaussian_neighbourhood((1, 1), (2, 2), 0.0), 0.0);
    }

    #[test]
    fn test_invalid_inputs() {
        let data = make_data();
        // Zero grid
        assert!(fit(&data, (0, 4), 10, 0.5, 2.0).is_err());
        // n_iter = 0
        assert!(fit(&data, (4, 4), 0, 0.5, 2.0).is_err());
        // bad lr
        assert!(fit(&data, (4, 4), 10, -0.1, 2.0).is_err());
    }
}
