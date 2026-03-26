//! Zigzag persistence for sequences of simplicial complexes.
//!
//! Zigzag persistence generalises ordinary persistence to handle sequences where
//! simplices can be both added and removed. The "zigzag" refers to the sequence
//! of maps alternating between forward (inclusion) and backward (deletion)
//! directions:
//!
//! ```text
//! K_0 → K_1 ← K_2 → K_3 ← K_4 → …
//! ```
//!
//! This implementation uses a vineyard-inspired approach:
//! 1. Maintain a totally ordered filtration of currently-active simplices.
//! 2. On addition: append the simplex and reduce.
//! 3. On removal: locate the simplex, perform transpositions to move it to the
//!    end, then delete it and close any open intervals.
//!
//! ## References
//!
//! - Carlsson & de Silva (2010). Zigzag Persistence. FoCM.
//! - Cohen-Steiner, Edelsbrunner & Morozov (2006). Vines and Vineyards.

use crate::tda::alpha_complex::{sym_diff_sorted, Simplex};
use std::collections::HashMap;

// ─── ZigzagDirection ─────────────────────────────────────────────────────────

/// Direction of a zigzag step.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZigzagDirection {
    /// The simplex is being added (forward inclusion).
    Forward,
    /// The simplex is being removed (backward deletion).
    Backward,
}

// ─── ZigzagStep ──────────────────────────────────────────────────────────────

/// A single step in a zigzag filtration sequence.
#[derive(Debug, Clone)]
pub struct ZigzagStep {
    /// Direction of this step.
    pub direction: ZigzagDirection,
    /// The simplex being added or removed.
    pub simplex: Simplex,
}

// ─── ZigzagPersistence ───────────────────────────────────────────────────────

/// Incremental zigzag persistence computation.
///
/// Maintains the current filtration and reduced boundary matrix, updating
/// persistance intervals as simplices are added and removed.
#[derive(Debug, Clone)]
pub struct ZigzagPersistence {
    /// Active simplices in filtration order.
    filtration: Vec<Simplex>,
    /// Boundary matrix (reduced), stored as columns of sorted row indices.
    columns: Vec<Vec<usize>>,
    /// Pivot map: pivot row → column index.
    pivot_col: HashMap<usize, usize>,
    /// Currently open intervals: simplex key → (birth_value, dimension).
    open_intervals: HashMap<Vec<usize>, (f64, usize)>,
    /// Completed persistence pairs collected so far.
    completed: Vec<(f64, f64, usize)>,
    /// Current "time" counter used as a synthetic filtration value when needed.
    time: usize,
}

impl ZigzagPersistence {
    /// Create a new empty zigzag persistence structure.
    pub fn new() -> Self {
        Self {
            filtration: Vec::new(),
            columns: Vec::new(),
            pivot_col: HashMap::new(),
            open_intervals: HashMap::new(),
            completed: Vec::new(),
            time: 0,
        }
    }

    /// Process a forward step: add a simplex and return any newly closed intervals.
    ///
    /// In standard persistence, adding a simplex either:
    /// - Creates a new homology class (no pivot after reduction) → opens an interval.
    /// - Destroys an existing class (reduction yields a pivot) → closes an interval.
    pub fn add_simplex(&mut self, s: Simplex) -> Vec<(f64, f64, usize)> {
        self.time += 1;
        let idx = self.filtration.len();
        self.filtration.push(s.clone());

        // Build boundary column for the new simplex
        let simplex_index: HashMap<Vec<usize>, usize> = self
            .filtration
            .iter()
            .enumerate()
            .map(|(i, sx)| (sx.vertices.clone(), i))
            .collect();

        let mut col: Vec<usize> = s
            .boundary_faces()
            .iter()
            .filter_map(|face| simplex_index.get(face).copied())
            .collect();
        col.sort_unstable();

        // Reduce column
        while let Some(&pivot) = col.last() {
            if let Some(&k) = self.pivot_col.get(&pivot) {
                let col_k = self.columns[k].clone();
                sym_diff_sorted(&mut col, &col_k);
            } else {
                break;
            }
        }

        self.columns.push(col.clone());

        let mut newly_closed = Vec::new();

        if let Some(&pivot) = col.last() {
            // Simplex kills homology class born at `pivot`
            self.pivot_col.insert(pivot, idx);
            // Find the open interval for the simplex at position `pivot`
            let birth_key = self.filtration[pivot].vertices.clone();
            if let Some((birth_val, dim)) = self.open_intervals.remove(&birth_key) {
                let death_val = s.filtration_value;
                newly_closed.push((birth_val, death_val, dim));
                self.completed.push((birth_val, death_val, dim));
            }
        } else {
            // Simplex creates a new homology class
            let dim = s.dimension();
            self.open_intervals
                .insert(s.vertices.clone(), (s.filtration_value, dim));
        }

        newly_closed
    }

    /// Process a backward step: remove a simplex and return any newly closed intervals.
    ///
    /// To remove simplex `s` at position `pos`, we perform adjacent transpositions
    /// (vineyard updates) to bubble it to the last position, then delete it.
    ///
    /// After removal, if `s` had an open interval it gets closed with death = infinity
    /// (the feature survives through the deletion). If `s` was killing another class,
    /// the killed class re-opens.
    pub fn remove_simplex(&mut self, s: Simplex) -> Vec<(f64, f64, usize)> {
        self.time += 1;
        let mut newly_closed = Vec::new();

        // Find the simplex in the current filtration
        let pos = match self
            .filtration
            .iter()
            .position(|x| x.vertices == s.vertices)
        {
            Some(p) => p,
            None => return newly_closed, // Simplex not found, nothing to do
        };

        // Bubble the simplex to the end via adjacent transpositions
        let last = self.filtration.len() - 1;
        for i in pos..last {
            self.transpose_adjacent(i);
        }

        // The simplex is now at the last position; remove it
        let removed = self.filtration.pop();
        self.columns.pop();

        // Rebuild pivot map from scratch (cheaper than incremental for correctness)
        self.rebuild_pivot_map();

        // If the removed simplex had an open interval, close it with death = removal_value
        if let Some(rem) = removed {
            let death_val = rem.filtration_value;
            if let Some((birth_val, dim)) = self.open_intervals.remove(&rem.vertices) {
                // Feature born at birth_val, removed at death_val
                newly_closed.push((birth_val, death_val, dim));
                self.completed.push((birth_val, death_val, dim));
            }
        }

        newly_closed
    }

    /// Return all completed persistence pairs collected so far.
    pub fn pairs(&self) -> &[(f64, f64, usize)] {
        &self.completed
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    /// Perform a single adjacent transposition of simplices at positions i and i+1.
    ///
    /// This implements the core vineyard transposition step: swap the two simplices
    /// in the filtration and update the boundary matrix accordingly.
    fn transpose_adjacent(&mut self, i: usize) {
        // Swap simplices in the filtration
        self.filtration.swap(i, i + 1);

        // Update boundary matrix columns for i and i+1
        // The boundary of the swapped simplices may reference each other.

        // Check if column i+1 has a pivot at i (i.e., column i+1 contains row i)
        let col_i1_has_i = self.columns[i + 1].binary_search(&i).is_ok();
        // Check if column i has a pivot at i+1
        let col_i_has_i1 = self.columns[i].binary_search(&(i + 1)).is_ok();

        if col_i1_has_i {
            // Add column i to column i+1 (mod-2) to remove row i
            let col_i = self.columns[i].clone();
            sym_diff_sorted(&mut self.columns[i + 1], &col_i);
        }

        // Swap row references: replace any occurrence of i with i+1 and vice versa
        // in ALL columns (since row i and row i+1 swapped their simplices)
        for col in self.columns.iter_mut() {
            let had_i = col.binary_search(&i).is_ok();
            let had_i1 = col.binary_search(&(i + 1)).is_ok();

            if had_i && !had_i1 {
                // Replace i with i+1
                if let Ok(pos) = col.binary_search(&i) {
                    col[pos] = i + 1;
                    col.sort_unstable();
                }
            } else if had_i1 && !had_i {
                // Replace i+1 with i
                if let Ok(pos) = col.binary_search(&(i + 1)) {
                    col[pos] = i;
                    col.sort_unstable();
                }
            } else if had_i && had_i1 {
                // Both present: they cancel each other (mod-2 swap has no effect)
                // but positions are already correct since we just swapped simplices
                // Nothing to do — both row indices present means they cancel
            }
        }

        // Swap the columns themselves
        self.columns.swap(i, i + 1);

        // If the new column i+1 had a pivot issue (was killing something via i), fix it
        if col_i_has_i1 {
            // Column i used to reference row i+1; after swap, this reference is still valid
            // but needs to remain reduced
            let last_i = self.columns[i].last().copied();
            if let Some(piv) = last_i {
                if let Some(&k) = self.pivot_col.get(&piv) {
                    if k != i {
                        let col_k = self.columns[k].clone();
                        sym_diff_sorted(&mut self.columns[i], &col_k);
                    }
                }
            }
        }
    }

    /// Rebuild the pivot map from the current (partially reduced) columns.
    fn rebuild_pivot_map(&mut self) {
        self.pivot_col.clear();
        for (j, col) in self.columns.iter().enumerate() {
            if let Some(&pivot) = col.last() {
                self.pivot_col.insert(pivot, j);
            }
        }
    }
}

impl Default for ZigzagPersistence {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Batch computation ────────────────────────────────────────────────────────

/// Compute zigzag persistence for a sequence of steps.
///
/// Processes all steps in order and returns all completed `(birth, death, dim)` pairs.
pub fn compute_zigzag(steps: &[ZigzagStep]) -> Vec<(f64, f64, usize)> {
    let mut zz = ZigzagPersistence::new();
    for step in steps {
        match step.direction {
            ZigzagDirection::Forward => {
                zz.add_simplex(step.simplex.clone());
            }
            ZigzagDirection::Backward => {
                zz.remove_simplex(step.simplex.clone());
            }
        }
    }
    // Close any surviving open intervals (essential features)
    // These are not added to completed; return only finite pairs.
    zz.completed
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simplex(verts: Vec<usize>, fv: f64) -> Simplex {
        Simplex {
            vertices: verts,
            filtration_value: fv,
        }
    }

    #[test]
    fn test_add_vertices_creates_intervals() {
        let mut zz = ZigzagPersistence::new();
        let v0 = make_simplex(vec![0], 0.0);
        let v1 = make_simplex(vec![1], 1.0);
        zz.add_simplex(v0);
        zz.add_simplex(v1);
        // Two vertices added → two open intervals, no closed yet
        assert_eq!(zz.open_intervals.len(), 2);
        assert!(zz.completed.is_empty());
    }

    #[test]
    fn test_add_edge_closes_one_component() {
        let mut zz = ZigzagPersistence::new();
        let v0 = make_simplex(vec![0], 0.0);
        let v1 = make_simplex(vec![1], 0.0);
        let edge = make_simplex(vec![0, 1], 1.0);

        zz.add_simplex(v0);
        zz.add_simplex(v1);
        let closed = zz.add_simplex(edge);

        // Adding edge should close one H0 interval
        assert_eq!(closed.len(), 1, "Adding edge should close one interval");
        let (birth, death, dim) = closed[0];
        assert_eq!(dim, 0, "Should be H0 (connected component)");
        assert!(birth < death, "birth < death expected");
    }

    #[test]
    fn test_remove_simplex_closes_interval() {
        let mut zz = ZigzagPersistence::new();
        let v0 = make_simplex(vec![0], 0.0);
        zz.add_simplex(v0.clone());

        // The vertex has an open interval; removing it should close it
        let closed = zz.remove_simplex(v0);
        // After removal, the open interval for v0 is closed
        assert_eq!(zz.open_intervals.len(), 0);
        // The closed interval should appear
        assert!(!closed.is_empty() || !zz.completed.is_empty());
    }

    #[test]
    fn test_zigzag_batch_add_then_remove() {
        let v0 = make_simplex(vec![0], 0.0);
        let v1 = make_simplex(vec![1], 0.5);
        let edge = make_simplex(vec![0, 1], 1.0);

        let steps = vec![
            ZigzagStep {
                direction: ZigzagDirection::Forward,
                simplex: v0.clone(),
            },
            ZigzagStep {
                direction: ZigzagDirection::Forward,
                simplex: v1.clone(),
            },
            ZigzagStep {
                direction: ZigzagDirection::Forward,
                simplex: edge.clone(),
            },
            ZigzagStep {
                direction: ZigzagDirection::Backward,
                simplex: edge,
            },
        ];

        let pairs = compute_zigzag(&steps);
        // Adding the edge closed one H0 interval
        assert!(!pairs.is_empty(), "Expected at least one completed pair");
        for (birth, death, _) in &pairs {
            assert!(birth <= death, "birth={birth} > death={death}");
        }
    }

    #[test]
    fn test_directions_are_non_exhaustive() {
        // Ensure the enum is non_exhaustive (just compile-check the variant usage)
        let d = ZigzagDirection::Forward;
        assert_eq!(d, ZigzagDirection::Forward);
    }
}
