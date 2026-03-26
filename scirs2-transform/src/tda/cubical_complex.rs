//! Cubical complex for image and volumetric grid data.
//!
//! A cubical complex represents topological features in rectilinear grid data
//! (images, voxels). Cells correspond to vertices (0-cells), edges (1-cells),
//! faces/pixels (2-cells), and voxels (3-cells).
//!
//! The filtration value of a cell is the **maximum** of the filtration values
//! of all its vertices (sup-level set filtration), enabling persistent homology
//! computation that correctly captures connected components, loops, and voids
//! in the data.
//!
//! ## References
//!
//! - Kaczynski, Mischaikow & Mrozek (2004). Computational Homology.
//! - Wagner, Chen & Vuçini (2011). Efficient Computation of Persistent Homology for Cubical Data.

use crate::error::{Result, TransformError};
use crate::tda::alpha_complex::sym_diff_sorted;
use std::collections::HashMap;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for cubical complex construction.
#[derive(Debug, Clone, Default)]
pub struct CubicalConfig {
    // Reserved for future extension.
}

// ─── CubicalCell ──────────────────────────────────────────────────────────────

/// A single cell in a cubical complex.
///
/// Cells are identified by their grid coordinates and dimension.
/// - 0-cell (vertex): coordinates are integer grid indices `[x, y]`
/// - 1-cell (edge): one coordinate is half-integer (encoded as doubled + 1)
/// - 2-cell (face): two coordinates are half-integer
/// - 3-cell (voxel): three coordinates are half-integer
///
/// To avoid floating point, we use "doubled coordinates": each coordinate is
/// stored as `2*k` for integers (vertices) and `2*k+1` for half-integers (edge midpoints).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CubicalCell {
    /// Doubled coordinates: even entries are integer, odd entries are half-integer.
    pub coordinates: Vec<usize>,
    /// Dimension of the cell (number of coordinates that are odd / half-integer).
    pub dimension: usize,
}

impl CubicalCell {
    /// Construct a cell from doubled coordinates (validity is checked by counting odd entries).
    pub fn new(coordinates: Vec<usize>) -> Self {
        let dimension = coordinates.iter().filter(|&&c| c % 2 == 1).count();
        Self {
            coordinates,
            dimension,
        }
    }

    /// Return the lexicographic sort key: (dimension, coordinates).
    pub fn sort_key(&self) -> (usize, &[usize]) {
        (self.dimension, &self.coordinates)
    }
}

// ─── CubicalComplex ───────────────────────────────────────────────────────────

/// A filtered cubical complex.
///
/// Filtration values are stored per cell and sorted in non-decreasing order.
#[derive(Debug, Clone)]
pub struct CubicalComplex {
    /// Sorted list of (cell, filtration_value).
    pub cells: Vec<(CubicalCell, f64)>,
    /// Spatial dimension of the complex (2 for images, 3 for volumes).
    pub spatial_dim: usize,
}

impl CubicalComplex {
    /// Build a 2D cubical complex from a 2D image (rows × cols).
    ///
    /// Filtration value of each cell = max pixel value over all vertices of that cell
    /// (sup-level set filtration). Cells:
    /// - 0-cells: `rows+1` × `cols+1` vertices
    /// - 1-cells: horizontal and vertical edges between adjacent vertices
    /// - 2-cells: `rows` × `cols` faces (pixel cells)
    pub fn from_image_2d(image: &[Vec<f64>]) -> Result<Self> {
        if image.is_empty() || image[0].is_empty() {
            return Err(TransformError::InvalidInput(
                "Image must be non-empty".to_string(),
            ));
        }
        let rows = image.len();
        let cols = image[0].len();
        for row in image.iter() {
            if row.len() != cols {
                return Err(TransformError::InvalidInput(
                    "All image rows must have the same length".to_string(),
                ));
            }
        }

        // Pixel value at (r, c) — vertex at (r, c) gets value = pixel(r, c)
        // We assign vertex values as pixel values directly.
        // For edges/faces, filtration = max over vertices.

        let vertex_value = |r: usize, c: usize| -> f64 {
            // Clamp to valid pixel range
            let pr = r.min(rows - 1);
            let pc = c.min(cols - 1);
            image[pr][pc]
        };

        let mut cells: Vec<(CubicalCell, f64)> = Vec::new();

        // 0-cells (vertices): (2r, 2c)
        for r in 0..=rows {
            for c in 0..=cols {
                let fv = vertex_value(r, c);
                cells.push((CubicalCell::new(vec![2 * r, 2 * c]), fv));
            }
        }

        // 1-cells — horizontal: (2r, 2c+1) connecting (2r,2c) and (2r,2c+2)
        for r in 0..=rows {
            for c in 0..cols {
                let fv = vertex_value(r, c).max(vertex_value(r, c + 1));
                cells.push((CubicalCell::new(vec![2 * r, 2 * c + 1]), fv));
            }
        }

        // 1-cells — vertical: (2r+1, 2c) connecting (2r,2c) and (2r+2,2c)
        for r in 0..rows {
            for c in 0..=cols {
                let fv = vertex_value(r, c).max(vertex_value(r + 1, c));
                cells.push((CubicalCell::new(vec![2 * r + 1, 2 * c]), fv));
            }
        }

        // 2-cells (pixels): (2r+1, 2c+1)
        for r in 0..rows {
            for c in 0..cols {
                let fv = vertex_value(r, c)
                    .max(vertex_value(r, c + 1))
                    .max(vertex_value(r + 1, c))
                    .max(vertex_value(r + 1, c + 1));
                cells.push((CubicalCell::new(vec![2 * r + 1, 2 * c + 1]), fv));
            }
        }

        // Sort by (filtration_value, dimension, coordinates)
        cells.sort_by(|(ca, fa), (cb, fb)| {
            fa.partial_cmp(fb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ca.dimension.cmp(&cb.dimension))
                .then(ca.coordinates.cmp(&cb.coordinates))
        });

        Ok(Self {
            cells,
            spatial_dim: 2,
        })
    }

    /// Build a 3D cubical complex from a 3D image (slices × rows × cols).
    pub fn from_image_3d(image: &[Vec<Vec<f64>>]) -> Result<Self> {
        if image.is_empty() || image[0].is_empty() || image[0][0].is_empty() {
            return Err(TransformError::InvalidInput(
                "3D image must be non-empty".to_string(),
            ));
        }
        let slices = image.len();
        let rows = image[0].len();
        let cols = image[0][0].len();

        let vertex_value = |s: usize, r: usize, c: usize| -> f64 {
            let ps = s.min(slices - 1);
            let pr = r.min(rows - 1);
            let pc = c.min(cols - 1);
            image[ps][pr][pc]
        };

        let mut cells: Vec<(CubicalCell, f64)> = Vec::new();

        // 0-cells
        for s in 0..=slices {
            for r in 0..=rows {
                for c in 0..=cols {
                    let fv = vertex_value(s, r, c);
                    cells.push((CubicalCell::new(vec![2 * s, 2 * r, 2 * c]), fv));
                }
            }
        }

        // 1-cells (3 orientations)
        // z-direction: (2s+1, 2r, 2c)
        for s in 0..slices {
            for r in 0..=rows {
                for c in 0..=cols {
                    let fv = vertex_value(s, r, c).max(vertex_value(s + 1, r, c));
                    cells.push((CubicalCell::new(vec![2 * s + 1, 2 * r, 2 * c]), fv));
                }
            }
        }
        // y-direction: (2s, 2r+1, 2c)
        for s in 0..=slices {
            for r in 0..rows {
                for c in 0..=cols {
                    let fv = vertex_value(s, r, c).max(vertex_value(s, r + 1, c));
                    cells.push((CubicalCell::new(vec![2 * s, 2 * r + 1, 2 * c]), fv));
                }
            }
        }
        // x-direction: (2s, 2r, 2c+1)
        for s in 0..=slices {
            for r in 0..=rows {
                for c in 0..cols {
                    let fv = vertex_value(s, r, c).max(vertex_value(s, r, c + 1));
                    cells.push((CubicalCell::new(vec![2 * s, 2 * r, 2 * c + 1]), fv));
                }
            }
        }

        // 2-cells (3 orientations)
        // zy-face: (2s+1, 2r+1, 2c)
        for s in 0..slices {
            for r in 0..rows {
                for c in 0..=cols {
                    let fv = vertex_value(s, r, c)
                        .max(vertex_value(s + 1, r, c))
                        .max(vertex_value(s, r + 1, c))
                        .max(vertex_value(s + 1, r + 1, c));
                    cells.push((CubicalCell::new(vec![2 * s + 1, 2 * r + 1, 2 * c]), fv));
                }
            }
        }
        // zx-face: (2s+1, 2r, 2c+1)
        for s in 0..slices {
            for r in 0..=rows {
                for c in 0..cols {
                    let fv = vertex_value(s, r, c)
                        .max(vertex_value(s + 1, r, c))
                        .max(vertex_value(s, r, c + 1))
                        .max(vertex_value(s + 1, r, c + 1));
                    cells.push((CubicalCell::new(vec![2 * s + 1, 2 * r, 2 * c + 1]), fv));
                }
            }
        }
        // yx-face: (2s, 2r+1, 2c+1)
        for s in 0..=slices {
            for r in 0..rows {
                for c in 0..cols {
                    let fv = vertex_value(s, r, c)
                        .max(vertex_value(s, r + 1, c))
                        .max(vertex_value(s, r, c + 1))
                        .max(vertex_value(s, r + 1, c + 1));
                    cells.push((CubicalCell::new(vec![2 * s, 2 * r + 1, 2 * c + 1]), fv));
                }
            }
        }

        // 3-cells (voxels): (2s+1, 2r+1, 2c+1)
        for s in 0..slices {
            for r in 0..rows {
                for c in 0..cols {
                    let fv = vertex_value(s, r, c)
                        .max(vertex_value(s + 1, r, c))
                        .max(vertex_value(s, r + 1, c))
                        .max(vertex_value(s, r, c + 1))
                        .max(vertex_value(s + 1, r + 1, c))
                        .max(vertex_value(s + 1, r, c + 1))
                        .max(vertex_value(s, r + 1, c + 1))
                        .max(vertex_value(s + 1, r + 1, c + 1));
                    cells.push((CubicalCell::new(vec![2 * s + 1, 2 * r + 1, 2 * c + 1]), fv));
                }
            }
        }

        cells.sort_by(|(ca, fa), (cb, fb)| {
            fa.partial_cmp(fb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ca.dimension.cmp(&cb.dimension))
                .then(ca.coordinates.cmp(&cb.coordinates))
        });

        Ok(Self {
            cells,
            spatial_dim: 3,
        })
    }

    /// Compute the boundary faces of a cubical cell.
    ///
    /// For a k-cell, the boundary consists of 2k faces obtained by decreasing
    /// or increasing each "active" (odd) coordinate by 1.
    pub fn boundary(&self, cell: &CubicalCell) -> Vec<CubicalCell> {
        if cell.dimension == 0 {
            return Vec::new();
        }
        let mut faces = Vec::new();
        for (i, &coord) in cell.coordinates.iter().enumerate() {
            if coord % 2 == 1 {
                // This coordinate is active (half-integer)
                // Decrease by 1
                let mut coords_low = cell.coordinates.clone();
                coords_low[i] = coord - 1;
                faces.push(CubicalCell::new(coords_low));
                // Increase by 1
                let mut coords_high = cell.coordinates.clone();
                coords_high[i] = coord + 1;
                faces.push(CubicalCell::new(coords_high));
            }
        }
        faces
    }

    /// Compute persistence diagram via boundary matrix reduction.
    ///
    /// Returns `(birth, death, dimension)` triples sorted by birth value.
    pub fn persistence_diagram(&self) -> Vec<(f64, f64, usize)> {
        let n = self.cells.len();

        // Map each cell to its index in the sorted filtration
        let cell_index: HashMap<&CubicalCell, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, (c, _))| (c, i))
            .collect();

        // Build boundary columns
        let mut columns: Vec<Vec<usize>> = self
            .cells
            .iter()
            .map(|(cell, _)| {
                let mut col: Vec<usize> = self
                    .boundary(cell)
                    .iter()
                    .filter_map(|face| cell_index.get(face).copied())
                    .collect();
                col.sort_unstable();
                col
            })
            .collect();

        // Standard reduction
        let mut pivot_col: HashMap<usize, usize> = HashMap::new();
        let mut pairs: Vec<(f64, f64, usize)> = Vec::new();

        for j in 0..n {
            while let Some(&pivot) = columns[j].last() {
                if let Some(&k) = pivot_col.get(&pivot) {
                    let col_k = columns[k].clone();
                    sym_diff_sorted(&mut columns[j], &col_k);
                } else {
                    break;
                }
            }

            if let Some(&pivot) = columns[j].last() {
                pivot_col.insert(pivot, j);
                let birth_idx = pivot;
                let death_idx = j;
                let (birth_cell, birth_fv) = &self.cells[birth_idx];
                let (_, death_fv) = &self.cells[death_idx];
                let dim = birth_cell.dimension;
                pairs.push((*birth_fv, *death_fv, dim));
            }
        }

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Count cells by dimension.
    pub fn cell_count(&self, dim: usize) -> usize {
        self.cells
            .iter()
            .filter(|(c, _)| c.dimension == dim)
            .count()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_3x3() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn test_cell_counts_3x3() {
        let img = simple_3x3();
        let cc = CubicalComplex::from_image_2d(&img).expect("should build");
        // rows=3, cols=3
        // 0-cells (vertices): (rows+1)*(cols+1) = 4*4 = 16
        // 1-cells horizontal: (rows+1)*cols = 4*3 = 12
        // 1-cells vertical:   rows*(cols+1) = 3*4 = 12
        // 2-cells (pixels):   rows*cols     = 3*3 = 9
        assert_eq!(cc.cell_count(0), 16, "Expected 16 vertices");
        assert_eq!(cc.cell_count(1), 24, "Expected 24 edges");
        assert_eq!(cc.cell_count(2), 9, "Expected 9 faces");
        assert_eq!(cc.cells.len(), 49, "Expected 49 total cells");
    }

    #[test]
    fn test_boundary_of_edge() {
        // Edge (0,1): doubled coords [0, 1] → boundary vertices [0,0] and [0,2]
        let edge = CubicalCell::new(vec![0, 1]);
        assert_eq!(edge.dimension, 1);
        let img = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let cc = CubicalComplex::from_image_2d(&img).expect("should build");
        let faces = cc.boundary(&edge);
        assert_eq!(faces.len(), 2);
        assert!(faces.iter().all(|f| f.dimension == 0));
    }

    #[test]
    fn test_boundary_of_vertex_empty() {
        let vtx = CubicalCell::new(vec![0, 0]);
        let img = vec![vec![1.0]];
        let cc = CubicalComplex::from_image_2d(&img).expect("should build");
        let faces = cc.boundary(&vtx);
        assert!(faces.is_empty());
    }

    #[test]
    fn test_persistence_diagram_non_trivial() {
        let img = simple_3x3();
        let cc = CubicalComplex::from_image_2d(&img).expect("should build");
        let diag = cc.persistence_diagram();
        // Should have at least one pair
        assert!(!diag.is_empty(), "Expected non-empty persistence diagram");
        for (birth, death, _) in &diag {
            assert!(birth <= death, "birth={birth} > death={death}");
        }
    }

    #[test]
    fn test_from_image_3d_basic() {
        let vol = vec![
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        ];
        let cc = CubicalComplex::from_image_3d(&vol).expect("3D build should succeed");
        // slices=2, rows=2, cols=2
        // 0-cells: 3*3*3 = 27
        assert_eq!(cc.cell_count(0), 27, "Expected 27 vertices");
        assert!(cc.cell_count(3) > 0, "Expected 3-cells");
    }

    #[test]
    fn test_invalid_empty_image() {
        let result = CubicalComplex::from_image_2d(&[]);
        assert!(result.is_err());
    }
}
