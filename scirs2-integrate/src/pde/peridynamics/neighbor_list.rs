//! Neighbor list construction using a spatial hash grid.
//!
//! A `BucketGrid` partitions 3-D space into axis-aligned cells of size equal to
//! the peridynamic horizon `delta`.  Finding all neighbors of a particle then
//! requires checking only the 3³ = 27 immediately adjacent cells, reducing the
//! cost from O(N²) to O(N) on average for uniform particle distributions.

use std::collections::HashMap;

/// A spatial hash grid for fast neighbor queries.
///
/// Particles are bucketed by integer cell coordinates derived from their
/// positions.  Cell size equals the peridynamic horizon so that all potential
/// neighbors reside in at most 27 cells.
#[derive(Debug, Clone)]
pub struct BucketGrid {
    /// Maps integer cell key `(ix, iy, iz)` to the list of particle indices in that cell.
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
    /// The peridynamic horizon (= cell side length).
    horizon: f64,
    /// Inverse horizon, cached for speed.
    inv_horizon: f64,
    /// Snapshot of particle positions used when building the grid.
    positions: Vec<[f64; 3]>,
}

impl BucketGrid {
    /// Build a bucket grid from a set of particle positions.
    ///
    /// # Arguments
    ///
    /// * `positions` - Slice of 3-D particle positions in the reference configuration.
    /// * `horizon`   - Peridynamic horizon radius δ.  Must be positive.
    pub fn new(positions: &[[f64; 3]], horizon: f64) -> Self {
        assert!(horizon > 0.0, "horizon must be positive");
        let inv_horizon = 1.0 / horizon;
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

        for (idx, pos) in positions.iter().enumerate() {
            let key = Self::cell_key(pos, inv_horizon);
            cells.entry(key).or_default().push(idx);
        }

        Self {
            cells,
            horizon,
            inv_horizon,
            positions: positions.to_vec(),
        }
    }

    /// Compute the integer cell key for a given position.
    #[inline]
    fn cell_key(pos: &[f64; 3], inv_horizon: f64) -> (i64, i64, i64) {
        (
            pos[0].mul_add(inv_horizon, 0.0).floor() as i64,
            pos[1].mul_add(inv_horizon, 0.0).floor() as i64,
            pos[2].mul_add(inv_horizon, 0.0).floor() as i64,
        )
    }

    /// Return all particle indices within the horizon of particle `i`.
    ///
    /// The result never contains `i` itself (no self-neighbors).
    pub fn find_neighbors(&self, i: usize, positions: &[[f64; 3]]) -> Vec<usize> {
        let ref_pos = &positions[i];
        let (cx, cy, cz) = Self::cell_key(ref_pos, self.inv_horizon);
        let h2 = self.horizon * self.horizon;

        let mut neighbors = Vec::new();
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(cell_particles) = self.cells.get(&key) {
                        for &j in cell_particles {
                            if j == i {
                                continue;
                            }
                            let p = &positions[j];
                            let dx2 = p[0] - ref_pos[0];
                            let dy2 = p[1] - ref_pos[1];
                            let dz2 = p[2] - ref_pos[2];
                            let dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                            if dist2 <= h2 {
                                neighbors.push(j);
                            }
                        }
                    }
                }
            }
        }
        neighbors
    }

    /// Return all particle indices within the horizon of a given query position.
    ///
    /// This is useful for inserting new particles or for off-particle queries.
    pub fn find_neighbors_at(&self, query: &[f64; 3]) -> Vec<usize> {
        let (cx, cy, cz) = Self::cell_key(query, self.inv_horizon);
        let h2 = self.horizon * self.horizon;

        let mut neighbors = Vec::new();
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(cell_particles) = self.cells.get(&key) {
                        for &j in cell_particles {
                            let p = &self.positions[j];
                            let dx2 = p[0] - query[0];
                            let dy2 = p[1] - query[1];
                            let dz2 = p[2] - query[2];
                            let dist2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                            if dist2 <= h2 {
                                neighbors.push(j);
                            }
                        }
                    }
                }
            }
        }
        neighbors
    }

    /// Return the horizon radius this grid was built with.
    #[inline]
    pub fn horizon(&self) -> f64 {
        self.horizon
    }

    /// Total number of particles stored in the grid.
    #[inline]
    pub fn n_particles(&self) -> usize {
        self.positions.len()
    }
}

/// Pre-computed neighbor list for all particles.
///
/// Stores the indices of all particles within the peridynamic horizon of each
/// particle.  The list is built once from the reference configuration and is
/// used to iterate over bonds efficiently during time integration.
#[derive(Debug, Clone)]
pub struct NeighborList {
    /// `neighbors[i]` contains all j such that |x_j - x_i| ≤ δ and j ≠ i.
    pub neighbors: Vec<Vec<usize>>,
    /// The peridynamic horizon used when building the list.
    pub horizon: f64,
}

impl NeighborList {
    /// Build a neighbor list from a set of reference positions.
    ///
    /// Uses `BucketGrid` for O(N) construction.
    pub fn build(positions: &[[f64; 3]], horizon: f64) -> Self {
        let grid = BucketGrid::new(positions, horizon);
        let n = positions.len();
        let mut neighbors = Vec::with_capacity(n);

        for i in 0..n {
            let nb = grid.find_neighbors(i, positions);
            neighbors.push(nb);
        }

        Self { neighbors, horizon }
    }

    /// Return the number of particles.
    #[inline]
    pub fn n_particles(&self) -> usize {
        self.neighbors.len()
    }

    /// Rebuild neighbor lists for particles whose bonds may have changed.
    ///
    /// In practice we rebuild the full grid; a more sophisticated approach
    /// would only update stale entries.
    pub fn rebuild(&mut self, positions: &[[f64; 3]]) {
        *self = Self::build(positions, self.horizon);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_particles(n: usize) -> Vec<[f64; 3]> {
        let mut positions = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    positions.push([
                        i as f64 / (n - 1) as f64,
                        j as f64 / (n - 1) as f64,
                        k as f64 / (n - 1) as f64,
                    ]);
                }
            }
        }
        positions
    }

    #[test]
    fn test_bucket_grid_no_self_neighbor() {
        let positions = unit_cube_particles(3);
        let grid = BucketGrid::new(&positions, 0.55);
        for i in 0..positions.len() {
            let nb = grid.find_neighbors(i, &positions);
            assert!(!nb.contains(&i), "particle {i} found itself as neighbor");
        }
    }

    #[test]
    fn test_bucket_grid_finds_all_within_horizon() {
        // Place particles at known positions
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.2, 0.0, 0.0], // outside horizon 0.1
        ];
        let grid = BucketGrid::new(&positions, 0.1);
        let nb0 = grid.find_neighbors(0, &positions);
        assert!(nb0.contains(&1), "particle 1 should be neighbor of 0");
        assert!(
            !nb0.contains(&2),
            "particle 2 outside horizon should not be neighbor"
        );
    }

    #[test]
    fn test_neighbor_list_build() {
        let positions = unit_cube_particles(3);
        let nl = NeighborList::build(&positions, 0.55);
        assert_eq!(nl.n_particles(), positions.len());
        // Center particle of a 3x3x3 grid at (0.5, 0.5, 0.5) should have many neighbors
        // Find center index
        let center_idx = positions.iter().position(|p| {
            (p[0] - 0.5).abs() < 1e-9 && (p[1] - 0.5).abs() < 1e-9 && (p[2] - 0.5).abs() < 1e-9
        });
        if let Some(ci) = center_idx {
            assert!(
                !nl.neighbors[ci].is_empty(),
                "center particle should have neighbors"
            );
        }
    }

    #[test]
    fn test_neighbor_list_particle_at_center_finds_sphere() {
        // 5x5x5 grid, spacing 0.1, horizon = 0.15 → should find 6 face neighbors
        let mut positions = Vec::new();
        for i in 0_usize..5 {
            for j in 0_usize..5 {
                for k in 0_usize..5 {
                    positions.push([i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1]);
                }
            }
        }
        let nl = NeighborList::build(&positions, 0.15);
        // Center at (0.2, 0.2, 0.2) = index 2*25 + 2*5 + 2 = 62
        let ci = 2 * 25 + 2 * 5 + 2;
        // 6 face neighbors at distance 0.1 and 12 edge at 0.1414 → both within 0.15
        // Only direct face neighbors (dist=0.1) should be within 0.15
        assert!(
            nl.neighbors[ci].len() >= 6,
            "expected >=6 neighbors, got {}",
            nl.neighbors[ci].len()
        );
    }
}
