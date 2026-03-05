//! Filtration (lens/filter) functions for the Mapper algorithm and persistence.
//!
//! A *filtration function* (also called a *lens* or *filter*) maps each data
//! point to a scalar value.  The Mapper algorithm uses these values to define
//! overlapping intervals that cover the image of the data, then clusters the
//! pre-image of each interval.
//!
//! # Available filtrations
//!
//! | Name | Description |
//! |------|-------------|
//! | [`EccentricityFiltration`] | L^p eccentricity: mean powered distance to all other points |
//! | [`DensityFiltration`] | Negative k-NN density estimate (so denser regions have lower values) |
//! | [`PcaFiltration`] | Projection onto the k-th principal component |
//! | [`LaplacianEigenvectorFiltration`] | k-th eigenvector of the normalised graph Laplacian |
//! | [`GeodesicDistanceFiltration`] | Approximate geodesic distance to a reference point via k-NN graph |
//!
//! All filtrations implement the [`Filtration`] trait and return an `Array1<f64>`
//! of length n (one value per data point).

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// A scalar function that maps each data point to a real value used as a
/// "lens" in the Mapper algorithm or as a height function for persistence.
pub trait Filtration: Send + Sync {
    /// Compute filter values for all n rows of `data` (shape n × d).
    ///
    /// Returns an `Array1<f64>` of length n.
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Eccentricity
// ─────────────────────────────────────────────────────────────────────────────

/// L^p eccentricity.
///
/// `e_p(x) = (1/n · Σ_y d(x,y)^p)^(1/p)`
///
/// For p = 1 this is the mean distance; for p = 2 it is the root mean square
/// distance.  High eccentricity values indicate "outlier" or boundary points.
#[derive(Debug, Clone)]
pub struct EccentricityFiltration {
    /// Exponent p (default: 1).
    pub p: f64,
}

impl Default for EccentricityFiltration {
    fn default() -> Self {
        Self { p: 1.0 }
    }
}

impl EccentricityFiltration {
    /// Create with the given exponent.
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl Filtration for EccentricityFiltration {
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if n == 0 || d == 0 {
            return Err(ClusteringError::InvalidInput(
                "eccentricity: data must be non-empty".into(),
            ));
        }

        let p = self.p;
        let mut ecc = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0_f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist_sq: f64 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                let dist = dist_sq.sqrt();
                sum += dist.powf(p);
            }
            let mean = sum / (n as f64);
            ecc[i] = mean.powf(1.0 / p);
        }
        Ok(ecc)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. k-NN Density
// ─────────────────────────────────────────────────────────────────────────────

/// k-nearest-neighbour density estimate.
///
/// `ρ_k(x) = k / (n · V_d · r_k(x)^d)`
///
/// where `r_k(x)` is the distance to the k-th nearest neighbour.  We return the
/// *negative* density so that denser regions receive lower filtration values
/// (useful for detecting peaks).
#[derive(Debug, Clone)]
pub struct DensityFiltration {
    /// Number of neighbours (default: 5).
    pub k: usize,
    /// If `true` return the negative density (default: false → return raw density).
    pub negate: bool,
}

impl Default for DensityFiltration {
    fn default() -> Self {
        Self { k: 5, negate: false }
    }
}

impl DensityFiltration {
    /// Create with given k and optional negation.
    pub fn new(k: usize, negate: bool) -> Self {
        Self { k, negate }
    }
}

impl Filtration for DensityFiltration {
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if n == 0 || d == 0 {
            return Err(ClusteringError::InvalidInput(
                "density filtration: data must be non-empty".into(),
            ));
        }
        let k = self.k.min(n - 1).max(1);

        let mut densities = Array1::zeros(n);
        for i in 0..n {
            // Collect distances from i to all other points.
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    data.row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let r_k = dists[k - 1].max(1e-15);

            // Volume of d-dimensional unit ball * r_k^d gives the volume.
            // For a simple density estimate we use 1/r_k^d.
            let vol = r_k.powi(d as i32);
            let density = (k as f64) / ((n as f64) * vol);
            densities[i] = if self.negate { -density } else { density };
        }
        Ok(densities)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. PCA projection
// ─────────────────────────────────────────────────────────────────────────────

/// Project data onto the k-th principal component (1-indexed, default k = 1).
///
/// Implements PCA via the power iteration / deflation method so that no
/// external LAPACK dependency is needed.
#[derive(Debug, Clone)]
pub struct PcaFiltration {
    /// Which principal component to use (1 = first / largest variance, etc.).
    pub component: usize,
    /// Maximum power-iteration steps.
    pub max_iter: usize,
}

impl Default for PcaFiltration {
    fn default() -> Self {
        Self {
            component: 1,
            max_iter: 200,
        }
    }
}

impl PcaFiltration {
    /// Create with a given component index (1-indexed).
    pub fn new(component: usize) -> Self {
        Self {
            component: component.max(1),
            max_iter: 200,
        }
    }
}

impl Filtration for PcaFiltration {
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if n < 2 || d == 0 {
            return Err(ClusteringError::InvalidInput(
                "PCA filtration: need at least 2 samples".into(),
            ));
        }

        // Centre the data.
        let mut centered = data.to_owned();
        for j in 0..d {
            let mean: f64 = centered.column(j).iter().sum::<f64>() / n as f64;
            for i in 0..n {
                centered[[i, j]] -= mean;
            }
        }

        // Power iteration to extract top-`component` eigenvectors.
        let k = self.component;
        let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(k);

        for pc_idx in 0..k {
            // Start with a random-ish vector (use a fixed seed pattern).
            let mut v: Vec<f64> = (0..d).map(|i| ((i + pc_idx + 1) as f64).sin()).collect();
            normalise_vec(&mut v);

            for _ in 0..self.max_iter {
                // Deflate by already-found eigenvectors.
                for ev in &eigenvecs {
                    let dot: f64 = v.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
                    for j in 0..d {
                        v[j] -= dot * ev[j];
                    }
                }

                // Multiply X^T X v (covariance matrix-vector product).
                // Mv = X^T (X v).
                let xv: Vec<f64> = (0..n)
                    .map(|i| {
                        centered
                            .row(i)
                            .iter()
                            .zip(v.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                    })
                    .collect();
                let mut mv = vec![0.0f64; d];
                for i in 0..n {
                    for j in 0..d {
                        mv[j] += centered[[i, j]] * xv[i];
                    }
                }

                // Deflate again.
                for ev in &eigenvecs {
                    let dot: f64 = mv.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
                    for j in 0..d {
                        mv[j] -= dot * ev[j];
                    }
                }

                let norm = normalise_vec(&mut mv);
                if norm < 1e-12 {
                    break;
                }

                // Check convergence.
                let change: f64 = v.iter().zip(mv.iter()).map(|(a, b)| (a - b).abs()).sum();
                v = mv;
                if change < 1e-8 {
                    break;
                }
            }
            eigenvecs.push(v);
        }

        // Project onto the requested component (last in eigenvecs).
        let ev = &eigenvecs[k - 1];
        let projections: Vec<f64> = (0..n)
            .map(|i| {
                centered
                    .row(i)
                    .iter()
                    .zip(ev.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        Array1::from_vec(projections)
            .into_shape_with_order(n)
            .map_err(|e| ClusteringError::ComputationError(e.to_string()))
    }
}

fn normalise_vec(v: &mut Vec<f64>) -> f64 {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Graph Laplacian Eigenvector
// ─────────────────────────────────────────────────────────────────────────────

/// k-nearest-neighbour graph Laplacian eigenvector filtration.
///
/// Builds a symmetric k-NN graph with Gaussian-kernel weights, then extracts
/// the k-th eigenvector of the normalised Laplacian via power iteration on the
/// deflated operator.  The second eigenvector (Fiedler vector) captures the
/// primary structural split of the data.
#[derive(Debug, Clone)]
pub struct LaplacianEigenvectorFiltration {
    /// Number of neighbours for the k-NN graph.
    pub k_neighbors: usize,
    /// Which Laplacian eigenvector to use (1 = Fiedler / smallest non-zero).
    pub component: usize,
    /// Gaussian kernel bandwidth parameter.
    pub sigma: Option<f64>,
    /// Maximum power-iteration steps.
    pub max_iter: usize,
}

impl Default for LaplacianEigenvectorFiltration {
    fn default() -> Self {
        Self {
            k_neighbors: 7,
            component: 1,
            sigma: None,
            max_iter: 300,
        }
    }
}

impl LaplacianEigenvectorFiltration {
    /// Create with given parameters.
    pub fn new(k_neighbors: usize, component: usize) -> Self {
        Self {
            k_neighbors,
            component,
            sigma: None,
            max_iter: 300,
        }
    }
}

impl Filtration for LaplacianEigenvectorFiltration {
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if n < 3 || d == 0 {
            return Err(ClusteringError::InvalidInput(
                "Laplacian eigenvector: need ≥ 3 samples".into(),
            ));
        }

        let k = self.k_neighbors.min(n - 1).max(1);

        // Build pairwise squared-distance matrix (needed for sigma estimation too).
        let mut sq_dists: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                sq_dists[i][j] = d2;
                sq_dists[j][i] = d2;
            }
        }

        // Estimate sigma from median k-NN distance if not provided.
        let sigma_sq = match self.sigma {
            Some(s) => s * s,
            None => {
                let mut knn_dists: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let mut row: Vec<f64> = sq_dists[i].clone();
                    row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    knn_dists.push(row[k].sqrt());
                }
                knn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = knn_dists[n / 2];
                (median * median).max(1e-10)
            }
        };

        // Build symmetric weight matrix W (Gaussian on k-NN graph).
        let mut w = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let mut sorted_idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            sorted_idx
                .sort_by(|&a, &b| sq_dists[i][a].partial_cmp(&sq_dists[i][b]).unwrap_or(std::cmp::Ordering::Equal));
            for &j in sorted_idx.iter().take(k) {
                let wij = (-sq_dists[i][j] / sigma_sq).exp();
                w[i][j] = wij;
                w[j][i] = wij;
            }
        }

        // Degree matrix D and normalised Laplacian: L_sym = D^{-1/2} L D^{-1/2}.
        let deg: Vec<f64> = (0..n).map(|i| w[i].iter().sum::<f64>().max(1e-15)).collect();
        let d_inv_sqrt: Vec<f64> = deg.iter().map(|&di| 1.0 / di.sqrt()).collect();

        // We want the *smallest* eigenvectors of L_sym except the trivial one.
        // Equivalently, the *largest* eigenvectors of I - L_sym = D^{-1/2} W D^{-1/2}.
        // Use power iteration on (I - L_sym) with deflation.

        let matvec = |v: &[f64]| -> Vec<f64> {
            // Compute (I - L_sym) v = D^{-1/2} W D^{-1/2} v.
            let scaled: Vec<f64> = v.iter().zip(d_inv_sqrt.iter()).map(|(vi, di)| vi * di).collect();
            let mut result = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += w[i][j] * scaled[j];
                }
                result[i] = d_inv_sqrt[i] * s;
            }
            result
        };

        let comp = self.component + 1; // +1 because we skip the trivial eigenvector.
        let mut eigenvecs: Vec<Vec<f64>> = Vec::with_capacity(comp);

        for pc_idx in 0..comp {
            let mut v: Vec<f64> = (0..n)
                .map(|i| ((i + pc_idx + 1) as f64).sin())
                .collect();
            let mut v_norm = vec_norm(&v);
            if v_norm > 1e-12 {
                for x in v.iter_mut() {
                    *x /= v_norm;
                }
            }

            for _ in 0..self.max_iter {
                let mut mv = matvec(&v);
                // Deflate.
                for ev in &eigenvecs {
                    let dot: f64 = mv.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
                    for j in 0..n {
                        mv[j] -= dot * ev[j];
                    }
                }
                v_norm = vec_norm(&mv);
                if v_norm < 1e-12 {
                    break;
                }
                for x in mv.iter_mut() {
                    *x /= v_norm;
                }
                let change: f64 = v.iter().zip(mv.iter()).map(|(a, b)| (a - b).abs()).sum();
                v = mv;
                if change < 1e-8 {
                    break;
                }
            }
            eigenvecs.push(v);
        }

        // Return the last eigenvector (corresponding to requested component).
        let ev = &eigenvecs[comp - 1];
        Ok(Array1::from_vec(ev.clone()))
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Geodesic Distance
// ─────────────────────────────────────────────────────────────────────────────

/// Approximate geodesic distance to a reference point via a k-NN graph and
/// Dijkstra's shortest-path algorithm.
///
/// This is a common lens for revealing manifold structure; the resulting filter
/// values are approximate geodesic distances from the reference point.
#[derive(Debug, Clone)]
pub struct GeodesicDistanceFiltration {
    /// Number of neighbours for the graph.
    pub k_neighbors: usize,
    /// Reference point index (default: 0).
    pub reference: usize,
}

impl Default for GeodesicDistanceFiltration {
    fn default() -> Self {
        Self {
            k_neighbors: 7,
            reference: 0,
        }
    }
}

impl GeodesicDistanceFiltration {
    /// Create with the given number of neighbours and reference point.
    pub fn new(k_neighbors: usize, reference: usize) -> Self {
        Self {
            k_neighbors,
            reference,
        }
    }
}

impl Filtration for GeodesicDistanceFiltration {
    fn apply(&self, data: ArrayView2<f64>) -> Result<Array1<f64>> {
        let n = data.nrows();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "geodesic filtration: data must be non-empty".into(),
            ));
        }
        if self.reference >= n {
            return Err(ClusteringError::InvalidInput(format!(
                "geodesic filtration: reference index {} out of range (n={})",
                self.reference, n
            )));
        }

        let k = self.k_neighbors.min(n - 1).max(1);

        // Build k-NN graph (adjacency list with edge weights = Euclidean distance).
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, d) in dists.into_iter().take(k) {
                adj[i].push((j, d));
                adj[j].push((i, d)); // symmetric
            }
        }

        // Dijkstra from `self.reference`.
        let mut dist = vec![f64::MAX; n];
        dist[self.reference] = 0.0;

        // Min-heap: (distance, node).
        let mut heap: BinaryHeap<DijkstraEntry> = BinaryHeap::new();
        heap.push(DijkstraEntry {
            neg_dist: 0.0,
            node: self.reference,
        });

        while let Some(DijkstraEntry { neg_dist, node }) = heap.pop() {
            let d = -neg_dist;
            if d > dist[node] {
                continue; // stale entry
            }
            for &(nb, w) in &adj[node] {
                let nd = d + w;
                if nd < dist[nb] {
                    dist[nb] = nd;
                    heap.push(DijkstraEntry {
                        neg_dist: -nd,
                        node: nb,
                    });
                }
            }
        }

        // Replace unreachable nodes with a large sentinel value.
        let max_finite = dist
            .iter()
            .copied()
            .filter(|&x| x < f64::MAX)
            .fold(0.0f64, f64::max);
        let result: Vec<f64> = dist
            .into_iter()
            .map(|x| if x == f64::MAX { max_finite * 2.0 } else { x })
            .collect();

        Ok(Array1::from_vec(result))
    }
}

use std::collections::BinaryHeap;

#[derive(Debug, PartialEq)]
struct DijkstraEntry {
    neg_dist: f64, // stored negated so BinaryHeap works as min-heap
    node: usize,
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_dist
            .partial_cmp(&other.neg_dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_blobs() -> Array2<f64> {
        // 6 points: 3 near (0,0) and 3 near (10,10).
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 9.9, 9.9, 10.1,
            ],
        )
        .expect("shape ok")
    }

    #[test]
    fn test_eccentricity_two_blobs() {
        let data = two_blobs();
        let filt = EccentricityFiltration::default();
        let vals = filt.apply(data.view()).expect("ok");
        assert_eq!(vals.len(), 6);
        // All values should be positive.
        for &v in vals.iter() {
            assert!(v > 0.0, "expected positive eccentricity, got {v}");
        }
    }

    #[test]
    fn test_density_two_blobs() {
        let data = two_blobs();
        let filt = DensityFiltration::new(2, false);
        let vals = filt.apply(data.view()).expect("ok");
        assert_eq!(vals.len(), 6);
        // Density should be positive.
        for &v in vals.iter() {
            assert!(v > 0.0, "expected positive density, got {v}");
        }
    }

    #[test]
    fn test_pca_projection() {
        let data = two_blobs();
        let filt = PcaFiltration::new(1);
        let vals = filt.apply(data.view()).expect("ok");
        assert_eq!(vals.len(), 6);
        // The two blobs should have very different PCA projection values.
        let blob0: f64 = vals.iter().take(3).copied().sum::<f64>() / 3.0;
        let blob1: f64 = vals.iter().skip(3).copied().sum::<f64>() / 3.0;
        assert!(
            (blob0 - blob1).abs() > 1.0,
            "PCA should separate blobs: blob0={blob0:.3} blob1={blob1:.3}"
        );
    }

    #[test]
    fn test_geodesic_distance() {
        let data = two_blobs();
        let filt = GeodesicDistanceFiltration::new(2, 0);
        let vals = filt.apply(data.view()).expect("ok");
        assert_eq!(vals.len(), 6);
        // Distance from point 0 to itself is 0.
        assert!(vals[0].abs() < 1e-10, "self-distance should be 0");
        // Points in the same blob should be close.
        assert!(vals[1] < 1.0, "within-blob geodesic should be small");
    }

    #[test]
    fn test_laplacian_eigenvector() {
        let data = two_blobs();
        let filt = LaplacianEigenvectorFiltration::new(3, 1);
        let vals = filt.apply(data.view()).expect("ok");
        assert_eq!(vals.len(), 6);
    }
}
