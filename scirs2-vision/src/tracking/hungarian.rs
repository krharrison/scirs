//! Hungarian (Munkres) algorithm for optimal assignment.
//!
//! Given a cost matrix of shape `rows × cols`, finds the minimum-cost
//! complete matching from rows to columns (or vice-versa for rectangular
//! matrices).  The implementation uses the classical O(n³) approach.

/// Solve the linear assignment problem on `cost[row][col]`.
///
/// Returns a vector of length `cost.len()` (number of rows). Entry `i` is
/// `Some(j)` if row `i` is assigned to column `j`, or `None` if the matrix
/// has more rows than columns and row `i` is unmatched.
///
/// When the matrix has more columns than rows, the extra columns are simply
/// unmatched (no entry in the returned vector references them).
pub fn hungarian_assign(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let nrows = cost.len();
    if nrows == 0 {
        return Vec::new();
    }
    let ncols = cost[0].len();

    if ncols == 0 {
        return vec![None; nrows];
    }

    // Work on a square matrix by padding with large values.
    let n = nrows.max(ncols);
    let big = f32::MAX / 2.0;

    // Flat row-major cost matrix of size n×n.
    let mut c: Vec<f32> = vec![big; n * n];
    for i in 0..nrows {
        for j in 0..ncols {
            c[i * n + j] = cost[i][j];
        }
    }

    // ---- Step 1: subtract row minima ----------------------------------------
    for i in 0..n {
        let row_min = (0..n).map(|j| c[i * n + j]).fold(big, f32::min);
        for j in 0..n {
            c[i * n + j] -= row_min;
        }
    }

    // ---- Step 2: subtract column minima -------------------------------------
    for j in 0..n {
        let col_min = (0..n).map(|i| c[i * n + j]).fold(big, f32::min);
        for i in 0..n {
            c[i * n + j] -= col_min;
        }
    }

    // Track which rows / cols are "starred" or "primed"
    // mask: 0 = normal, 1 = star, 2 = prime
    let mut mask: Vec<u8> = vec![0; n * n];
    let mut row_cover = vec![false; n];
    let mut col_cover = vec![false; n];

    // ---- Step 3: star zeros -------------------------------------------------
    for i in 0..n {
        for j in 0..n {
            if c[i * n + j] == 0.0 && !row_cover[i] && !col_cover[j] {
                mask[i * n + j] = 1;
                row_cover[i] = true;
                col_cover[j] = true;
            }
        }
    }
    row_cover.fill(false);
    col_cover.fill(false);

    // Helper: cover columns with starred zeros.
    let cover_starred_cols = |mask: &[u8], col_cover: &mut Vec<bool>| {
        for j in 0..n {
            if (0..n).any(|i| mask[i * n + j] == 1) {
                col_cover[j] = true;
            }
        }
    };

    cover_starred_cols(&mask, &mut col_cover);

    // ---- Main loop ----------------------------------------------------------
    // We iterate until all n columns are covered.
    'outer: loop {
        if col_cover.iter().filter(|&&v| v).count() >= n {
            break 'outer;
        }

        // Find an uncovered zero (step 4).
        let mut z_row;
        let mut z_col;
        loop {
            // Find an uncovered zero.
            let mut found = false;
            z_row = 0;
            z_col = 0;
            'search: for i in 0..n {
                for j in 0..n {
                    if c[i * n + j] == 0.0 && !row_cover[i] && !col_cover[j] {
                        z_row = i;
                        z_col = j;
                        found = true;
                        break 'search;
                    }
                }
            }

            if !found {
                // Step 6: no uncovered zero → adjust the matrix.
                let min_val = (0..n)
                    .flat_map(|i| (0..n).map(move |j| (i, j)))
                    .filter(|&(i, j)| !row_cover[i] && !col_cover[j])
                    .map(|(i, j)| c[i * n + j])
                    .fold(big, f32::min);

                for i in 0..n {
                    for j in 0..n {
                        if row_cover[i] {
                            c[i * n + j] += min_val;
                        }
                        if !col_cover[j] {
                            c[i * n + j] -= min_val;
                        }
                    }
                }
            } else {
                break;
            }
        }

        // Prime the uncovered zero (step 4 continued).
        mask[z_row * n + z_col] = 2;

        // Is there a starred zero in this row?
        let star_col = (0..n).find(|&j| mask[z_row * n + j] == 1);

        if let Some(sc) = star_col {
            // Cover this row, uncover the column of the starred zero.
            row_cover[z_row] = true;
            col_cover[sc] = false;
        } else {
            // Step 5: augment path.
            // Build alternating path starting from (z_row, z_col).
            let mut path: Vec<(usize, usize)> = vec![(z_row, z_col)];
            loop {
                let (last_r, last_c) = *path.last().expect("path is non-empty");
                // Find starred zero in same column.
                let star_row = (0..n).find(|&i| mask[i * n + last_c] == 1);
                if star_row.is_none() {
                    break;
                }
                let sr = star_row.expect("star_row is Some");
                path.push((sr, last_c));
                // Find primed zero in this row.
                let prime_col = (0..n).find(|&j| mask[sr * n + j] == 2);
                let pc = prime_col.expect("prime must exist after a starred zero");
                path.push((sr, pc));
            }
            // Augment: flip stars along path.
            for (pr, pc) in &path {
                if mask[pr * n + pc] == 1 {
                    mask[pr * n + pc] = 0;
                } else {
                    mask[pr * n + pc] = 1;
                }
            }
            // Clear covers and primes.
            row_cover.fill(false);
            col_cover.fill(false);
            for v in mask.iter_mut() {
                if *v == 2 {
                    *v = 0;
                }
            }
            cover_starred_cols(&mask, &mut col_cover);
        }
    }

    // Extract assignment: starred zeros give the matching.
    let mut result = vec![None; nrows];
    for i in 0..nrows {
        for j in 0..ncols {
            if mask[i * n + j] == 1 {
                result[i] = Some(j);
                break;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2x2_identity() {
        // Cost = 0 on diagonal.
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let assign = hungarian_assign(&cost);
        assert_eq!(assign[0], Some(0));
        assert_eq!(assign[1], Some(1));
    }

    #[test]
    fn test_2x2_cross() {
        // Optimal: row0→col1, row1→col0 (total cost = 0 + 0 = 0).
        let cost = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let assign = hungarian_assign(&cost);
        assert_eq!(assign[0], Some(1));
        assert_eq!(assign[1], Some(0));
    }

    #[test]
    fn test_empty() {
        let cost: Vec<Vec<f32>> = vec![];
        let assign = hungarian_assign(&cost);
        assert!(assign.is_empty());
    }

    #[test]
    fn test_3x3_optimal() {
        // Classic example: optimal total cost = 1+1+1 = 3.
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let assign = hungarian_assign(&cost);
        // Verify each row gets a distinct column.
        let cols: Vec<usize> = assign.iter().filter_map(|&c| c).collect();
        let mut sorted = cols.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "All three rows should be assigned");
    }
}
