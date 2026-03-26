//! Run-Length Matrix (RLM / GLRLM) Computation and Feature Extraction
//!
//! The Gray-Level Run-Length Matrix (GLRLM) characterizes texture by counting
//! runs of consecutive pixels sharing the same gray level along a specified
//! direction. A "run" is a maximal sequence of collinear pixels all having the
//! same quantized gray level.
//!
//! The matrix `R[i, j]` gives the number of runs with gray level `i` and
//! run length `j + 1`.
//!
//! # Directions
//!
//! Four scan directions are supported: 0 deg (horizontal), 45 deg, 90 deg (vertical),
//! and 135 deg.
//!
//! # Features
//!
//! From the RLM the following features are extracted:
//! - Short Run Emphasis (SRE)
//! - Long Run Emphasis (LRE)
//! - Gray-Level Non-Uniformity (GLN)
//! - Run-Length Non-Uniformity (RLN)
//! - Run Percentage (RP)
//! - Low Gray-Level Run Emphasis (LGRE)
//! - High Gray-Level Run Emphasis (HGRE)
//!
//! # References
//!
//! - Galloway, M.M. (1975). "Texture analysis using gray level run lengths."
//!   Computer Graphics and Image Processing, 4(2), 172-179.
//! - Chu, A., Sehgal, C.M., Greenleaf, J.F. (1990). "Use of gray value
//!   distribution of run lengths for texture analysis." Pattern Recognition
//!   Letters, 11(6), 415-419.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;

/// Scan direction for run-length computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlmDirection {
    /// Horizontal (0 deg): left-to-right scan
    Horizontal,
    /// Diagonal 45 deg: lower-left to upper-right
    Diagonal45,
    /// Vertical (90 deg): top-to-bottom scan
    Vertical,
    /// Diagonal 135 deg: upper-left to lower-right
    Diagonal135,
}

/// Features extracted from a Run-Length Matrix.
#[derive(Debug, Clone)]
pub struct RlmFeatures {
    /// Short Run Emphasis: emphasizes short runs.
    /// `SRE = (1/n_runs) * sum_j sum_i R(i,j) / (j+1)^2`
    pub short_run_emphasis: f64,
    /// Long Run Emphasis: emphasizes long runs.
    /// `LRE = (1/n_runs) * sum_j sum_i R(i,j) * (j+1)^2`
    pub long_run_emphasis: f64,
    /// Gray-Level Non-Uniformity: measures the variation of gray-level distribution.
    /// `GLN = (1/n_runs) * sum_i (sum_j R(i,j))^2`
    pub gray_level_non_uniformity: f64,
    /// Run-Length Non-Uniformity: measures the variation of run-length distribution.
    /// `RLN = (1/n_runs) * sum_j (sum_i R(i,j))^2`
    pub run_length_non_uniformity: f64,
    /// Run Percentage: ratio of total runs to total pixels.
    /// `RP = n_runs / n_pixels`
    pub run_percentage: f64,
    /// Low Gray-Level Run Emphasis: emphasizes runs at low gray levels.
    /// `LGRE = (1/n_runs) * sum_j sum_i R(i,j) / (i+1)^2`
    pub low_gray_level_run_emphasis: f64,
    /// High Gray-Level Run Emphasis: emphasizes runs at high gray levels.
    /// `HGRE = (1/n_runs) * sum_j sum_i R(i,j) * (i+1)^2`
    pub high_gray_level_run_emphasis: f64,
}

/// Combined result from RLM computation.
#[derive(Debug, Clone)]
pub struct RlmResult {
    /// The run-length matrix of shape `(n_levels, max_run_length)`.
    pub matrix: Array2<f64>,
    /// Extracted features.
    pub features: RlmFeatures,
}

/// Compute the Gray-Level Run-Length Matrix from an 8-bit image.
///
/// `image` must contain values in `[0, n_levels - 1]`. Values >= `n_levels`
/// are clamped to `n_levels - 1`.
///
/// # Parameters
/// - `image` - 8-bit grayscale image
/// - `direction` - scan direction
/// - `n_levels` - number of gray levels (>= 2)
///
/// # Returns
/// `Array2<f64>` of shape `(n_levels, max_run_length)` where `max_run_length`
/// is the longest run found.
///
/// # Errors
/// Returns error if `n_levels < 2` or image is empty.
pub fn compute_rlm(
    image: &Array2<u8>,
    direction: RlmDirection,
    n_levels: usize,
) -> NdimageResult<Array2<f64>> {
    if n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }

    let runs = collect_runs(image, direction, rows, cols, n_levels);

    // Determine maximum run length
    let max_run = runs.iter().map(|&(_, l)| l).max().unwrap_or(1);

    let mut matrix = Array2::<f64>::zeros((n_levels, max_run));
    for &(g, l) in &runs {
        matrix[[g, l - 1]] += 1.0;
    }

    Ok(matrix)
}

/// Extract RLM features from a pre-computed run-length matrix.
///
/// # Parameters
/// - `rlm` - Run-length matrix of shape `(n_levels, max_run_length)`
/// - `n_pixels` - total number of pixels in the image (rows * cols)
///
/// # Errors
/// Returns error if the matrix is empty or `n_pixels == 0`.
pub fn rlm_features(rlm: &Array2<f64>, n_pixels: usize) -> NdimageResult<RlmFeatures> {
    let (n_levels, max_run) = rlm.dim();
    if n_levels == 0 || max_run == 0 {
        return Err(NdimageError::InvalidInput("RLM must be non-empty".into()));
    }
    if n_pixels == 0 {
        return Err(NdimageError::InvalidInput("n_pixels must be > 0".into()));
    }

    let n_runs: f64 = rlm.iter().sum();
    if n_runs < 1e-15 {
        // No runs at all - return zero features
        return Ok(RlmFeatures {
            short_run_emphasis: 0.0,
            long_run_emphasis: 0.0,
            gray_level_non_uniformity: 0.0,
            run_length_non_uniformity: 0.0,
            run_percentage: 0.0,
            low_gray_level_run_emphasis: 0.0,
            high_gray_level_run_emphasis: 0.0,
        });
    }

    let mut sre = 0.0f64;
    let mut lre = 0.0f64;
    let mut lgre = 0.0f64;
    let mut hgre = 0.0f64;

    for i in 0..n_levels {
        for j in 0..max_run {
            let r = rlm[[i, j]];
            if r < 1e-15 {
                continue;
            }
            let run_len = (j + 1) as f64;
            let gray = (i + 1) as f64;

            sre += r / (run_len * run_len);
            lre += r * run_len * run_len;
            lgre += r / (gray * gray);
            hgre += r * gray * gray;
        }
    }

    // Gray-Level Non-Uniformity
    let mut gln = 0.0f64;
    for i in 0..n_levels {
        let row_sum: f64 = (0..max_run).map(|j| rlm[[i, j]]).sum();
        gln += row_sum * row_sum;
    }

    // Run-Length Non-Uniformity
    let mut rln = 0.0f64;
    for j in 0..max_run {
        let col_sum: f64 = (0..n_levels).map(|i| rlm[[i, j]]).sum();
        rln += col_sum * col_sum;
    }

    Ok(RlmFeatures {
        short_run_emphasis: sre / n_runs,
        long_run_emphasis: lre / n_runs,
        gray_level_non_uniformity: gln / n_runs,
        run_length_non_uniformity: rln / n_runs,
        run_percentage: n_runs / n_pixels as f64,
        low_gray_level_run_emphasis: lgre / n_runs,
        high_gray_level_run_emphasis: hgre / n_runs,
    })
}

// ---------------------------------------------------------------------------
// Internal: run collection
// ---------------------------------------------------------------------------

/// Collect `(gray_level, run_length)` pairs for all runs.
fn collect_runs(
    image: &Array2<u8>,
    direction: RlmDirection,
    rows: usize,
    cols: usize,
    n_levels: usize,
) -> Vec<(usize, usize)> {
    let lines = scan_lines(direction, rows, cols);
    let mut runs = Vec::new();

    for line in &lines {
        if line.is_empty() {
            continue;
        }
        let mut idx = 0;
        while idx < line.len() {
            let (r, c) = line[idx];
            let g = (image[[r, c]] as usize).min(n_levels - 1);
            let start = idx;
            while idx < line.len() {
                let (rr, cc) = line[idx];
                if (image[[rr, cc]] as usize).min(n_levels - 1) != g {
                    break;
                }
                idx += 1;
            }
            runs.push((g, idx - start));
        }
    }

    runs
}

/// Generate all scan lines for a given direction.
fn scan_lines(direction: RlmDirection, rows: usize, cols: usize) -> Vec<Vec<(usize, usize)>> {
    match direction {
        RlmDirection::Horizontal => (0..rows)
            .map(|r| (0..cols).map(|c| (r, c)).collect())
            .collect(),
        RlmDirection::Vertical => (0..cols)
            .map(|c| (0..rows).map(|r| (r, c)).collect())
            .collect(),
        RlmDirection::Diagonal45 => {
            // 45 deg: row decreases, col increases (lower-left to upper-right)
            // Starting points: bottom row left-to-right, then left column bottom-to-top
            let mut lines = Vec::new();
            // Start from bottom row
            for c0 in 0..cols {
                let line: Vec<(usize, usize)> = (0..)
                    .map(|k| (rows as i64 - 1 - k, c0 as i64 + k))
                    .take_while(|&(r, c)| r >= 0 && c < cols as i64)
                    .map(|(r, c)| (r as usize, c as usize))
                    .collect();
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            // Start from left column (excluding bottom-left corner, already covered)
            for r0 in (0..rows - 1).rev() {
                let line: Vec<(usize, usize)> = (0..)
                    .map(|k| (r0 as i64 - k, k))
                    .take_while(|&(r, c)| r >= 0 && c < cols as i64)
                    .map(|(r, c)| (r as usize, c as usize))
                    .collect();
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            lines
        }
        RlmDirection::Diagonal135 => {
            // 135 deg: row increases, col increases (upper-left to lower-right)
            let mut lines = Vec::new();
            // Start from top row
            for c0 in 0..cols {
                let line: Vec<(usize, usize)> = (0..)
                    .map(|k| (k as i64, c0 as i64 + k as i64))
                    .take_while(|&(r, c)| r < rows as i64 && c < cols as i64)
                    .map(|(r, c)| (r as usize, c as usize))
                    .collect();
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            // Start from left column (excluding top-left corner, already covered)
            for r0 in 1..rows {
                let line: Vec<(usize, usize)> = (0..)
                    .map(|k| (r0 as i64 + k as i64, k as i64))
                    .take_while(|&(r, c)| r < rows as i64 && c < cols as i64)
                    .map(|(r, c)| (r as usize, c as usize))
                    .collect();
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            lines
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_rlm_uniform_image() {
        // Uniform image: all pixels are the same gray level.
        // All runs should have the maximum length (equal to the row/col count).
        let img = Array2::from_elem((4, 6), 2u8);
        let rlm = compute_rlm(&img, RlmDirection::Horizontal, 4).expect("rlm");
        // 4 rows, each a single run of length 6 at gray=2
        assert_eq!(rlm.dim(), (4, 6));
        assert!(
            (rlm[[2, 5]] - 4.0).abs() < 1e-10,
            "Expected 4 runs of length 6"
        );

        let feats = rlm_features(&rlm, 24).expect("features");
        // All runs are the same length => RLN should be high
        // All runs are the same gray level => GLN should be high
        assert!(feats.run_percentage > 0.0);
        assert!(feats.short_run_emphasis > 0.0);
        assert!(feats.long_run_emphasis > 0.0);
    }

    #[test]
    fn test_rlm_checkerboard() {
        // Checkerboard pattern: every pixel differs from its neighbor.
        // All horizontal runs should be length 1.
        let img = Array2::from_shape_fn((4, 4), |(i, j)| ((i + j) % 2) as u8);
        let rlm = compute_rlm(&img, RlmDirection::Horizontal, 2).expect("rlm");
        // All runs are length 1
        assert_eq!(rlm.dim(), (2, 1));
        let total: f64 = rlm.iter().sum();
        assert!((total - 16.0).abs() < 1e-10, "Should have 16 unit runs");

        let feats = rlm_features(&rlm, 16).expect("features");
        // SRE should be 1.0 (all runs are length 1, so each contributes 1/1^2 = 1)
        assert!(
            (feats.short_run_emphasis - 1.0).abs() < 1e-10,
            "SRE should be 1.0 for all unit runs, got {}",
            feats.short_run_emphasis
        );
        // LRE should also be 1.0 (all length 1, so 1^2 = 1)
        assert!(
            (feats.long_run_emphasis - 1.0).abs() < 1e-10,
            "LRE should be 1.0 for all unit runs, got {}",
            feats.long_run_emphasis
        );
        // RP = 16/16 = 1.0
        assert!(
            (feats.run_percentage - 1.0).abs() < 1e-10,
            "RP should be 1.0, got {}",
            feats.run_percentage
        );
    }

    #[test]
    fn test_rlm_vertical_runs() {
        // Vertical stripes: each column is a single gray level
        let img = Array2::from_shape_fn((6, 4), |(_, j)| j as u8);
        let rlm = compute_rlm(&img, RlmDirection::Vertical, 4).expect("rlm");
        // Each column has a single vertical run of length 6
        assert!(rlm.dim().1 >= 6);
        for g in 0..4 {
            assert!(
                (rlm[[g, 5]] - 1.0).abs() < 1e-10,
                "Each gray level should have 1 vertical run of length 6"
            );
        }
    }

    #[test]
    fn test_rlm_diagonal_directions() {
        let img = Array2::from_elem((5, 5), 0u8);
        let rlm_45 = compute_rlm(&img, RlmDirection::Diagonal45, 2).expect("rlm 45");
        let rlm_135 = compute_rlm(&img, RlmDirection::Diagonal135, 2).expect("rlm 135");

        // Uniform image: all diagonals are runs of the same length
        let total_45: f64 = rlm_45.iter().sum();
        let total_135: f64 = rlm_135.iter().sum();
        // Both should have the same number of total runs (diagonals)
        assert!(
            (total_45 - total_135).abs() < 1e-10,
            "Uniform image should have same run count in both diagonal dirs"
        );
        // 5x5 image has 9 diagonals
        assert!(
            (total_45 - 9.0).abs() < 1e-10,
            "5x5 image should have 9 diagonal runs, got {}",
            total_45
        );
    }

    #[test]
    fn test_rlm_known_image() {
        // Known 3x3 image:
        // 0 0 1
        // 0 1 1
        // 1 1 1
        let img = Array2::from_shape_vec((3, 3), vec![0, 0, 1, 0, 1, 1, 1, 1, 1]).expect("ok");
        let rlm = compute_rlm(&img, RlmDirection::Horizontal, 2).expect("rlm");
        // Row 0: run(0,2) run(1,1) => R[0,1]+=1, R[1,0]+=1
        // Row 1: run(0,1) run(1,2) => R[0,0]+=1, R[1,1]+=1
        // Row 2: run(1,3)          => R[1,2]+=1
        assert!((rlm[[0, 1]] - 1.0).abs() < 1e-10); // gray=0, length=2
        assert!((rlm[[0, 0]] - 1.0).abs() < 1e-10); // gray=0, length=1
        assert!((rlm[[1, 0]] - 1.0).abs() < 1e-10); // gray=1, length=1
        assert!((rlm[[1, 1]] - 1.0).abs() < 1e-10); // gray=1, length=2
        assert!((rlm[[1, 2]] - 1.0).abs() < 1e-10); // gray=1, length=3
    }

    #[test]
    fn test_rlm_features_bounds() {
        let img = Array2::from_shape_fn((8, 8), |(i, j)| ((i * 8 + j) % 4) as u8);
        let rlm = compute_rlm(&img, RlmDirection::Horizontal, 4).expect("rlm");
        let feats = rlm_features(&rlm, 64).expect("features");

        // All features should be non-negative
        assert!(feats.short_run_emphasis >= 0.0);
        assert!(feats.long_run_emphasis >= 0.0);
        assert!(feats.gray_level_non_uniformity >= 0.0);
        assert!(feats.run_length_non_uniformity >= 0.0);
        assert!(feats.run_percentage > 0.0 && feats.run_percentage <= 1.0);
        assert!(feats.low_gray_level_run_emphasis >= 0.0);
        assert!(feats.high_gray_level_run_emphasis >= 0.0);
    }

    #[test]
    fn test_rlm_errors() {
        let img = Array2::from_elem((4, 4), 0u8);
        assert!(compute_rlm(&img, RlmDirection::Horizontal, 1).is_err());

        let empty = Array2::<u8>::zeros((0, 0));
        assert!(compute_rlm(&empty, RlmDirection::Horizontal, 2).is_err());

        let rlm = Array2::<f64>::zeros((0, 0));
        assert!(rlm_features(&rlm, 16).is_err());
    }
}
