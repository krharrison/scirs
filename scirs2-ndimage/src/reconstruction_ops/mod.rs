//! Morphological reconstruction operations.
//!
//! This module provides reconstruction by dilation/erosion, opening/closing by
//! reconstruction, hole filling, and top-hat by reconstruction.
//!
//! These operations are shape-preserving: they grow/shrink the marker image
//! subject to a mask constraint, iterating until stability.

use crate::error::{NdimageError, NdimageResult};
use std::collections::VecDeque;

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// 4-connected neighbours of pixel (r, c) within (rows, cols).
#[inline]
fn neighbours4(r: usize, c: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut nb = Vec::with_capacity(4);
    if r > 0       { nb.push((r - 1, c)); }
    if r + 1 < rows { nb.push((r + 1, c)); }
    if c > 0       { nb.push((r, c - 1)); }
    if c + 1 < cols { nb.push((r, c + 1)); }
    nb.into_iter()
}

/// 8-connected neighbours of pixel (r, c) within (rows, cols).
#[inline]
fn neighbours8(r: usize, c: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut nb = Vec::with_capacity(8);
    let r_min = if r > 0 { r - 1 } else { 0 };
    let r_max = (r + 1).min(rows - 1);
    let c_min = if c > 0 { c - 1 } else { 0 };
    let c_max = (c + 1).min(cols - 1);
    for nr in r_min..=r_max {
        for nc in c_min..=c_max {
            if nr != r || nc != c {
                nb.push((nr, nc));
            }
        }
    }
    nb.into_iter()
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Morphological reconstruction by dilation.
///
/// Iteratively dilates `marker` (clamped by `mask`) until stability.
/// Both arrays must have equal shape.
///
/// # Errors
/// Returns `DimensionError` if shapes differ.
pub fn reconstruction_by_dilation(
    marker: &[Vec<f64>],
    mask: &[Vec<f64>],
) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = marker.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = marker[0].len();
    // Validate shapes.
    if mask.len() != rows || mask.iter().any(|r| r.len() != cols)
        || marker.iter().any(|r| r.len() != cols)
    {
        return Err(NdimageError::DimensionError(
            "marker and mask must have the same shape".into(),
        ));
    }
    // Validate: marker ≤ mask everywhere.
    let mut result: Vec<Vec<f64>> = marker
        .iter()
        .zip(mask.iter())
        .map(|(mr, msr)| {
            mr.iter()
                .zip(msr.iter())
                .map(|(&m, &ms)| m.min(ms))
                .collect()
        })
        .collect();

    // Queue-based: seed queue with every pixel.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            queue.push_back((r, c));
        }
    }

    while let Some((r, c)) = queue.pop_front() {
        let val = result[r][c];
        for (nr, nc) in neighbours8(r, c, rows, cols) {
            let new_val = val.max(result[nr][nc]).min(mask[nr][nc]);
            if new_val > result[nr][nc] {
                result[nr][nc] = new_val;
                queue.push_back((nr, nc));
            }
        }
    }

    Ok(result)
}

/// Morphological reconstruction by erosion.
///
/// Iteratively erodes `marker` (clamped by `mask`) until stability.
/// Both arrays must have equal shape and `marker >= mask` initially.
///
/// # Errors
/// Returns `DimensionError` if shapes differ.
pub fn reconstruction_by_erosion(
    marker: &[Vec<f64>],
    mask: &[Vec<f64>],
) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = marker.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = marker[0].len();
    if mask.len() != rows || mask.iter().any(|r| r.len() != cols)
        || marker.iter().any(|r| r.len() != cols)
    {
        return Err(NdimageError::DimensionError(
            "marker and mask must have the same shape".into(),
        ));
    }
    // Clamp: marker >= mask.
    let mut result: Vec<Vec<f64>> = marker
        .iter()
        .zip(mask.iter())
        .map(|(mr, msr)| {
            mr.iter()
                .zip(msr.iter())
                .map(|(&m, &ms)| m.max(ms))
                .collect()
        })
        .collect();

    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            queue.push_back((r, c));
        }
    }

    while let Some((r, c)) = queue.pop_front() {
        let val = result[r][c];
        for (nr, nc) in neighbours8(r, c, rows, cols) {
            let new_val = val.min(result[nr][nc]).max(mask[nr][nc]);
            if new_val < result[nr][nc] {
                result[nr][nc] = new_val;
                queue.push_back((nr, nc));
            }
        }
    }

    Ok(result)
}

/// Opening by reconstruction.
///
/// Erodes `image` with a flat structuring element of given `se_size`, then
/// reconstructs by dilation.  Removes features smaller than the SE but
/// preserves the exact shape of surviving features.
///
/// # Arguments
/// * `image`   – input 2-D image as `Vec<Vec<f64>>`
/// * `erosion` – the result of applying morphological erosion to `image`
pub fn opening_by_reconstruction(
    image: &[Vec<f64>],
    erosion: &[Vec<f64>],
) -> NdimageResult<Vec<Vec<f64>>> {
    reconstruction_by_dilation(erosion, image)
}

/// Closing by reconstruction.
///
/// Dilates `image` then reconstructs by erosion.
///
/// # Arguments
/// * `image`    – input 2-D image
/// * `dilation` – the result of applying morphological dilation to `image`
pub fn closing_by_reconstruction(
    image: &[Vec<f64>],
    dilation: &[Vec<f64>],
) -> NdimageResult<Vec<Vec<f64>>> {
    reconstruction_by_erosion(dilation, image)
}

/// Fill holes in a binary image.
///
/// Equivalent to flood-filling from the border with the inverted image.
/// A "hole" is a connected region of `false` pixels that does not touch the
/// border.
///
/// Uses 4-connectivity for the background flood fill.
pub fn fill_holes(binary: &[Vec<bool>]) -> NdimageResult<Vec<Vec<bool>>> {
    let rows = binary.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = binary[0].len();
    if binary.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError(
            "binary image rows must have equal length".into(),
        ));
    }

    // background[r][c] = true when the pixel is background reachable from border.
    let mut background = vec![vec![false; cols]; rows];

    // Seed from border.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let seed_border = |r: usize, c: usize, bg: &mut Vec<Vec<bool>>, q: &mut VecDeque<(usize, usize)>| {
        if !binary[r][c] && !bg[r][c] {
            bg[r][c] = true;
            q.push_back((r, c));
        }
    };
    for c in 0..cols {
        seed_border(0, c, &mut background, &mut queue);
        seed_border(rows - 1, c, &mut background, &mut queue);
    }
    for r in 1..rows.saturating_sub(1) {
        seed_border(r, 0, &mut background, &mut queue);
        seed_border(r, cols - 1, &mut background, &mut queue);
    }

    // BFS over background (non-foreground) pixels.
    while let Some((r, c)) = queue.pop_front() {
        for (nr, nc) in neighbours4(r, c, rows, cols) {
            if !binary[nr][nc] && !background[nr][nc] {
                background[nr][nc] = true;
                queue.push_back((nr, nc));
            }
        }
    }

    // A pixel is filled if it is background but not border-reachable.
    let result = binary
        .iter()
        .enumerate()
        .map(|(r, row)| {
            row.iter()
                .enumerate()
                .map(|(c, &fg)| fg || !background[r][c])
                .collect::<Vec<bool>>()
        })
        .collect();

    Ok(result)
}

/// Top-hat by reconstruction (white top-hat).
///
/// Computes `image - opening_by_reconstruction(image, erode(image, se_size))`.
/// Highlights bright structures smaller than `se_size`.
///
/// # Arguments
/// * `image`   – input 2-D image
/// * `se_size` – half-size of the flat square structuring element (radius)
pub fn top_hat_by_reconstruction(
    image: &[Vec<f64>],
    se_size: usize,
) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError(
            "image rows must have equal length".into(),
        ));
    }

    // Flat erosion with a square SE of half-size se_size.
    let eroded = flat_erosion(image, se_size)?;
    // Opening by reconstruction.
    let opened = opening_by_reconstruction(image, &eroded)?;
    // Subtract.
    let result = image
        .iter()
        .zip(opened.iter())
        .map(|(ir, or_row)| {
            ir.iter().zip(or_row.iter()).map(|(&i, &o)| i - o).collect()
        })
        .collect();

    Ok(result)
}

// ─── Internal: flat morphological erosion ─────────────────────────────────────

/// Flat erosion with a square SE of `radius` in each direction (i.e., window
/// size 2*radius+1 × 2*radius+1), using a running-minimum approach.
fn flat_erosion(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = image[0].len();

    // Row-wise erosion first.
    let mut row_eroded = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let c_min = c.saturating_sub(radius);
            let c_max = (c + radius + 1).min(cols);
            let mut min_val = f64::INFINITY;
            for k in c_min..c_max {
                if image[r][k] < min_val {
                    min_val = image[r][k];
                }
            }
            row_eroded[r][c] = min_val;
        }
    }

    // Then column-wise erosion.
    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let r_min = r.saturating_sub(radius);
            let r_max = (r + radius + 1).min(rows);
            let mut min_val = f64::INFINITY;
            for k in r_min..r_max {
                if row_eroded[k][c] < min_val {
                    min_val = row_eroded[k][c];
                }
            }
            result[r][c] = min_val;
        }
    }

    Ok(result)
}

/// Flat dilation with a square SE of `radius` in each direction.
pub(crate) fn flat_dilation(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = image[0].len();

    let mut row_dilated = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let c_min = c.saturating_sub(radius);
            let c_max = (c + radius + 1).min(cols);
            let mut max_val = f64::NEG_INFINITY;
            for k in c_min..c_max {
                if image[r][k] > max_val {
                    max_val = image[r][k];
                }
            }
            row_dilated[r][c] = max_val;
        }
    }

    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let r_min = r.saturating_sub(radius);
            let r_max = (r + radius + 1).min(rows);
            let mut max_val = f64::NEG_INFINITY;
            for k in r_min..r_max {
                if row_dilated[k][c] > max_val {
                    max_val = row_dilated[k][c];
                }
            }
            result[r][c] = max_val;
        }
    }

    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(data: &[&[f64]]) -> Vec<Vec<f64>> {
        data.iter().map(|row| row.to_vec()).collect()
    }

    #[test]
    fn test_reconstruction_by_dilation_basic() {
        // Mask = marker when they are equal: result = marker.
        let m = make_image(&[&[0.0, 1.0, 0.0], &[1.0, 2.0, 1.0], &[0.0, 1.0, 0.0]]);
        let result = reconstruction_by_dilation(&m, &m).expect("recon failed");
        for (r, row) in result.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!((v - m[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_reconstruction_clamps_to_mask() {
        // marker > mask somewhere: after clamping, result ≤ mask.
        let marker = make_image(&[&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]]);
        let mask = make_image(&[&[1.0, 1.0, 1.0], &[1.0, 3.0, 1.0], &[1.0, 1.0, 1.0]]);
        let result = reconstruction_by_dilation(&marker, &mask).expect("recon failed");
        for (r, row) in result.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!(v <= mask[r][c] + 1e-12, "r={} c={}: {} > {}", r, c, v, mask[r][c]);
            }
        }
    }

    #[test]
    fn test_fill_holes_simple() {
        // A ring of foreground with a hole in the middle.
        #[rustfmt::skip]
        let binary = vec![
            vec![false, false, false, false, false],
            vec![false, true,  true,  true,  false],
            vec![false, true,  false, true,  false],
            vec![false, true,  true,  true,  false],
            vec![false, false, false, false, false],
        ];
        let filled = fill_holes(&binary).expect("fill_holes failed");
        // The hole at (2,2) should now be true.
        assert!(filled[2][2], "hole should be filled");
        // Border pixels are still false.
        assert!(!filled[0][0]);
        assert!(!filled[4][4]);
    }

    #[test]
    fn test_fill_holes_no_holes() {
        let binary = vec![
            vec![false, false, false],
            vec![false, true,  false],
            vec![false, false, false],
        ];
        let filled = fill_holes(&binary).expect("fill_holes failed");
        // No hole because the false pixels touch the border.
        assert!(!filled[0][0]);
        assert!(filled[1][1]);
    }

    #[test]
    fn test_top_hat_by_reconstruction() {
        // A flat background with a single bright pixel.
        let mut image = vec![vec![0.1f64; 20]; 20];
        image[10][10] = 1.0;
        let tophat = top_hat_by_reconstruction(&image, 2).expect("tophat failed");
        // The bright spot should produce a high top-hat value.
        assert!(tophat[10][10] > 0.5, "expected bright top-hat at centre");
        // Flat background → top-hat ≈ 0.
        assert!(tophat[0][0] < 0.1);
    }

    #[test]
    fn test_opening_by_reconstruction() {
        let mut image = vec![vec![0.0f64; 10]; 10];
        // Large feature.
        for r in 2..8 {
            for c in 2..8 {
                image[r][c] = 1.0;
            }
        }
        // Small feature.
        image[0][0] = 1.0;
        let eroded = flat_erosion(&image, 2).expect("erosion failed");
        let opened = opening_by_reconstruction(&image, &eroded).expect("opening failed");
        // Small feature should be removed.
        assert!(opened[0][0] < 0.5, "small feature should be removed by opening");
        // Large feature partially preserved.
        assert!(opened[4][4] > 0.5, "large feature should survive");
    }

    #[test]
    fn test_reconstruction_by_erosion_basic() {
        let mask = vec![vec![0.0f64; 5]; 5];
        let marker = vec![vec![1.0f64; 5]; 5];
        let result = reconstruction_by_erosion(&marker, &mask).expect("recon by erosion failed");
        for row in &result {
            for &v in row {
                assert!((v - 0.0).abs() < 1e-12, "expected all zeros");
            }
        }
    }
}
