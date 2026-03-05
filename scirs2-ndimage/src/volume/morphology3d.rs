//! 3D Morphological Operations
//!
//! Provides comprehensive 3-D morphological image processing:
//!
//! * [`erosion_3d`]               – grayscale erosion
//! * [`dilation_3d`]              – grayscale dilation
//! * [`opening_3d`]               – erosion followed by dilation
//! * [`closing_3d`]               – dilation followed by erosion
//! * [`binary_erosion_3d`]        – binary erosion (boolean volume)
//! * [`binary_dilation_3d`]       – binary dilation (boolean volume)
//! * [`connected_components_3d`]  – 26-connectivity connected-component labeling
//! * [`skeleton_3d`]              – 3-D topological thinning (skeletonization)
//!
//! All operations accept a [`StructElem3D`] structuring element.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array3, ArrayView3};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Structuring element
// ---------------------------------------------------------------------------

/// A 3-D structuring element for morphological operations.
#[derive(Debug, Clone)]
pub enum StructElem3D {
    /// Full cubic element; `side = 2 * radius + 1`.
    Cube(usize),
    /// Spherical element with given floating-point radius.
    Sphere(f64),
    /// 6-connectivity cross (face-adjacent only).
    Cross6,
    /// 26-connectivity: entire 3×3×3 cube.
    Cross26,
    /// Custom boolean volume (depth × height × width, center at floor(shape/2)).
    Custom(Array3<bool>),
}

impl StructElem3D {
    /// Convert to a dense boolean `Array3<bool>`.
    pub fn to_array(&self) -> Array3<bool> {
        match self {
            StructElem3D::Cube(radius) => {
                let s = 2 * radius + 1;
                Array3::from_elem((s, s, s), true)
            }
            StructElem3D::Sphere(r) => {
                let ri = r.ceil() as usize;
                let s = 2 * ri + 1;
                let cr = ri as f64;
                Array3::from_shape_fn((s, s, s), |(z, y, x)| {
                    let dz = z as f64 - cr;
                    let dy = y as f64 - cr;
                    let dx = x as f64 - cr;
                    (dz * dz + dy * dy + dx * dx).sqrt() <= *r
                })
            }
            StructElem3D::Cross6 => {
                let mut a = Array3::<bool>::from_elem((3, 3, 3), false);
                a[[1, 1, 1]] = true;
                a[[0, 1, 1]] = true;
                a[[2, 1, 1]] = true;
                a[[1, 0, 1]] = true;
                a[[1, 2, 1]] = true;
                a[[1, 1, 0]] = true;
                a[[1, 1, 2]] = true;
                a
            }
            StructElem3D::Cross26 => Array3::from_elem((3, 3, 3), true),
            StructElem3D::Custom(arr) => arr.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Border helper
// ---------------------------------------------------------------------------

#[inline]
fn get_bool(vol: &Array3<bool>, z: isize, y: isize, x: isize) -> bool {
    let sh = vol.shape();
    let in_bounds = z >= 0
        && y >= 0
        && x >= 0
        && (z as usize) < sh[0]
        && (y as usize) < sh[1]
        && (x as usize) < sh[2];
    if in_bounds {
        vol[[z as usize, y as usize, x as usize]]
    } else {
        false // out-of-bounds treated as background
    }
}

#[inline]
fn get_f64_clamped(vol: &Array3<f64>, z: isize, y: isize, x: isize) -> f64 {
    let sh = vol.shape();
    let zc = z.clamp(0, sh[0].saturating_sub(1) as isize) as usize;
    let yc = y.clamp(0, sh[1].saturating_sub(1) as isize) as usize;
    let xc = x.clamp(0, sh[2].saturating_sub(1) as isize) as usize;
    vol[[zc, yc, xc]]
}

// ---------------------------------------------------------------------------
// Grayscale erosion / dilation
// ---------------------------------------------------------------------------

/// Apply grayscale erosion to a 3-D volume.
///
/// At each voxel the output is the **minimum** value over all positions where
/// the structuring element is `true`.
pub fn erosion_3d(
    volume: ArrayView3<f64>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<f64>> {
    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let se = struct_element.to_array();
    let se_sh = se.shape();
    let (sze, sye, sxe) = (se_sh[0], se_sh[1], se_sh[2]);
    let hz = (sze / 2) as isize;
    let hy = (sye / 2) as isize;
    let hx = (sxe / 2) as isize;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut min_val = f64::INFINITY;
                for sz in 0..sze {
                    for sy in 0..sye {
                        for sx in 0..sxe {
                            if se[[sz, sy, sx]] {
                                let v = get_f64_clamped(
                                    &vol,
                                    z as isize + (sz as isize) - hz,
                                    y as isize + (sy as isize) - hy,
                                    x as isize + (sx as isize) - hx,
                                );
                                if v < min_val {
                                    min_val = v;
                                }
                            }
                        }
                    }
                }
                out[[z, y, x]] = if min_val.is_infinite() {
                    vol[[z, y, x]]
                } else {
                    min_val
                };
            }
        }
    }
    Ok(out)
}

/// Apply grayscale dilation to a 3-D volume.
///
/// At each voxel the output is the **maximum** value over all positions where
/// the structuring element is `true`.
pub fn dilation_3d(
    volume: ArrayView3<f64>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<f64>> {
    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let se = struct_element.to_array();
    let se_sh = se.shape();
    let (sze, sye, sxe) = (se_sh[0], se_sh[1], se_sh[2]);
    let hz = (sze / 2) as isize;
    let hy = (sye / 2) as isize;
    let hx = (sxe / 2) as isize;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut max_val = f64::NEG_INFINITY;
                for sz in 0..sze {
                    for sy in 0..sye {
                        for sx in 0..sxe {
                            if se[[sz, sy, sx]] {
                                let v = get_f64_clamped(
                                    &vol,
                                    z as isize + (sz as isize) - hz,
                                    y as isize + (sy as isize) - hy,
                                    x as isize + (sx as isize) - hx,
                                );
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                        }
                    }
                }
                out[[z, y, x]] = if max_val.is_infinite() {
                    vol[[z, y, x]]
                } else {
                    max_val
                };
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Opening / Closing
// ---------------------------------------------------------------------------

/// Apply morphological opening (erosion then dilation).
pub fn opening_3d(
    volume: ArrayView3<f64>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<f64>> {
    let eroded = erosion_3d(volume, struct_element)?;
    dilation_3d(eroded.view(), struct_element)
}

/// Apply morphological closing (dilation then erosion).
pub fn closing_3d(
    volume: ArrayView3<f64>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<f64>> {
    let dilated = dilation_3d(volume, struct_element)?;
    erosion_3d(dilated.view(), struct_element)
}

// ---------------------------------------------------------------------------
// Binary erosion / dilation
// ---------------------------------------------------------------------------

/// Apply binary erosion to a boolean 3-D volume.
///
/// A voxel is `true` in the output only when every structuring-element position
/// that is `true` overlaps a foreground voxel in the input.
pub fn binary_erosion_3d(
    volume: ArrayView3<bool>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<bool>> {
    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let se = struct_element.to_array();
    let se_sh = se.shape();
    let (sze, sye, sxe) = (se_sh[0], se_sh[1], se_sh[2]);
    let hz = (sze / 2) as isize;
    let hy = (sye / 2) as isize;
    let hx = (sxe / 2) as isize;
    let mut out = Array3::<bool>::from_elem((nz, ny, nx), false);

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut fits = true;
                'se_loop: for sz in 0..sze {
                    for sy in 0..sye {
                        for sx in 0..sxe {
                            if se[[sz, sy, sx]]
                                && !get_bool(
                                    &vol,
                                    z as isize + (sz as isize) - hz,
                                    y as isize + (sy as isize) - hy,
                                    x as isize + (sx as isize) - hx,
                                )
                            {
                                fits = false;
                                break 'se_loop;
                            }
                        }
                    }
                }
                out[[z, y, x]] = fits;
            }
        }
    }
    Ok(out)
}

/// Apply binary dilation to a boolean 3-D volume.
///
/// A voxel is `true` in the output when at least one structuring-element
/// position that is `true` overlaps a foreground voxel in the input.
pub fn binary_dilation_3d(
    volume: ArrayView3<bool>,
    struct_element: &StructElem3D,
) -> NdimageResult<Array3<bool>> {
    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let se = struct_element.to_array();
    let se_sh = se.shape();
    let (sze, sye, sxe) = (se_sh[0], se_sh[1], se_sh[2]);
    let hz = (sze / 2) as isize;
    let hy = (sye / 2) as isize;
    let hx = (sxe / 2) as isize;
    let mut out = Array3::<bool>::from_elem((nz, ny, nx), false);

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut hit = false;
                'se_loop: for sz in 0..sze {
                    for sy in 0..sye {
                        for sx in 0..sxe {
                            if se[[sz, sy, sx]]
                                && get_bool(
                                    &vol,
                                    z as isize + (sz as isize) - hz,
                                    y as isize + (sy as isize) - hy,
                                    x as isize + (sx as isize) - hx,
                                )
                            {
                                hit = true;
                                break 'se_loop;
                            }
                        }
                    }
                }
                out[[z, y, x]] = hit;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Connected component labeling (26-connectivity)
// ---------------------------------------------------------------------------

/// Label connected components in a binary 3-D volume using 26-connectivity.
///
/// Returns an `Array3<u32>` where `0` denotes background and positive values
/// are unique component labels.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty.
pub fn connected_components_3d(
    volume: ArrayView3<bool>,
) -> NdimageResult<(Array3<u32>, u32)> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let vol = volume.to_owned();
    let mut labels = Array3::<u32>::zeros((nz, ny, nx));
    let mut next_label: u32 = 1;

    // 26-connectivity offsets
    let offsets: Vec<(isize, isize, isize)> = (-1_isize..=1)
        .flat_map(|dz| {
            (-1_isize..=1).flat_map(move |dy| {
                (-1_isize..=1).filter_map(move |dx| {
                    if dz == 0 && dy == 0 && dx == 0 {
                        None
                    } else {
                        Some((dz, dy, dx))
                    }
                })
            })
        })
        .collect();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if vol[[z, y, x]] && labels[[z, y, x]] == 0 {
                    // BFS flood fill
                    labels[[z, y, x]] = next_label;
                    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
                    queue.push_back((z, y, x));

                    while let Some((cz, cy, cx)) = queue.pop_front() {
                        for &(dz, dy, dx) in &offsets {
                            let nz_i = cz as isize + dz;
                            let ny_i = cy as isize + dy;
                            let nx_i = cx as isize + dx;

                            if nz_i >= 0
                                && ny_i >= 0
                                && nx_i >= 0
                                && (nz_i as usize) < nz
                                && (ny_i as usize) < ny
                                && (nx_i as usize) < nx
                            {
                                let nz_u = nz_i as usize;
                                let ny_u = ny_i as usize;
                                let nx_u = nx_i as usize;
                                if vol[[nz_u, ny_u, nx_u]]
                                    && labels[[nz_u, ny_u, nx_u]] == 0
                                {
                                    labels[[nz_u, ny_u, nx_u]] = next_label;
                                    queue.push_back((nz_u, ny_u, nx_u));
                                }
                            }
                        }
                    }
                    next_label += 1;
                }
            }
        }
    }

    Ok((labels, next_label - 1))
}

// ---------------------------------------------------------------------------
// 3-D skeletonization (iterative thinning)
// ---------------------------------------------------------------------------

/// Compute the 3-D skeleton of a binary volume using iterative thinning.
///
/// The algorithm repeatedly removes "simple points" (points whose removal does
/// not change the topology of the binary volume) from the surface of the object
/// until no more such points exist.  Simple-point detection uses the 26-
/// connectivity Euler number criterion.
///
/// This is an approximation of the Palagyi–Kuba parallel thinning algorithm
/// adapted for correctness; it converges but may be slower than parallel
/// implementations.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty.
pub fn skeleton_3d(volume: ArrayView3<bool>) -> NdimageResult<Array3<bool>> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let mut cur = volume.to_owned();

    // 6-connected surface-detection offsets
    let face_offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let mut changed = true;
    while changed {
        changed = false;
        let prev = cur.clone();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if !prev[[z, y, x]] {
                        continue;
                    }
                    // Check that the point is a surface point (has at least one
                    // background 6-neighbor)
                    let is_surface = face_offsets.iter().any(|&(dz, dy, dx)| {
                        let nz_i = z as isize + dz;
                        let ny_i = y as isize + dy;
                        let nx_i = x as isize + dx;
                        let in_bounds = nz_i >= 0
                            && ny_i >= 0
                            && nx_i >= 0
                            && (nz_i as usize) < nz
                            && (ny_i as usize) < ny
                            && (nx_i as usize) < nx;
                        !in_bounds || !prev[[nz_i as usize, ny_i as usize, nx_i as usize]]
                    });

                    if !is_surface {
                        continue;
                    }

                    // Simple-point test: removing this voxel must not change
                    // 26-connectivity of the 3×3×3 neighbourhood.
                    if is_simple_point(&prev, z, y, x) {
                        cur[[z, y, x]] = false;
                        changed = true;
                    }
                }
            }
        }
    }
    Ok(cur)
}

/// Test whether removing the voxel at `(z, y, x)` is a topologically simple
/// operation within its 3×3×3 26-neighbourhood.
///
/// A point is *simple* iff:
///  1. It has at least one foreground 26-neighbour (not isolated).
///  2. After removal, the 26-connected foreground components in the
///     neighbourhood are unchanged in count.
///  3. The background remains 6-connected in the neighbourhood.
fn is_simple_point(vol: &Array3<bool>, z: usize, y: usize, x: usize) -> bool {
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);

    // Extract 3×3×3 neighbourhood (with boundary padding = background)
    let mut nbh = [false; 27]; // [z][y][x] in order
    let mut fg_count = 0_usize;

    for dz in -1_isize..=1 {
        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                let nz_i = z as isize + dz;
                let ny_i = y as isize + dy;
                let nx_i = x as isize + dx;
                let idx = ((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)) as usize;
                let v = if nz_i >= 0
                    && ny_i >= 0
                    && nx_i >= 0
                    && (nz_i as usize) < nz
                    && (ny_i as usize) < ny
                    && (nx_i as usize) < nx
                {
                    vol[[nz_i as usize, ny_i as usize, nx_i as usize]]
                } else {
                    false
                };
                nbh[idx] = v;
                if idx != 13 && v {
                    fg_count += 1; // count fg neighbours (exclude center)
                }
            }
        }
    }

    // Must have at least one foreground neighbour
    if fg_count == 0 {
        return false;
    }

    // Count 26-connected foreground components in nbh (excluding center)
    let n26_fg = count_26_components(&nbh, false);

    // Count 6-connected background components in nbh (including center as bg)
    let n6_bg = count_6_bg_components(&nbh);

    // Simple point: exactly 1 fg component, exactly 1 bg component
    n26_fg == 1 && n6_bg == 1
}

/// Count 26-connected components in a 3×3×3 boolean neighbourhood.
///
/// `exclude_center` – whether index 13 (the centre) is excluded.
fn count_26_components(nbh: &[bool; 27], exclude_center: bool) -> usize {
    let mut visited = [false; 27];
    let mut count = 0_usize;

    for start in 0..27_usize {
        if start == 13 && exclude_center {
            continue;
        }
        if !nbh[start] || visited[start] {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::<usize>::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(cur) = queue.pop_front() {
            let (cz, cy, cx) = (cur / 9, (cur % 9) / 3, cur % 3);
            for dz in -1_isize..=1 {
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let nz = cz as isize + dz;
                        let ny = cy as isize + dy;
                        let nx = cx as isize + dx;
                        if nz >= 0 && ny >= 0 && nx >= 0 && nz < 3 && ny < 3 && nx < 3 {
                            let ni = (nz * 9 + ny * 3 + nx) as usize;
                            if ni == 13 && exclude_center {
                                continue;
                            }
                            if nbh[ni] && !visited[ni] {
                                visited[ni] = true;
                                queue.push_back(ni);
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

/// Count 6-connected background components treating the centre as background.
fn count_6_bg_components(nbh: &[bool; 27]) -> usize {
    // Build a background mask (centre is always bg)
    let mut bg = [false; 27];
    for i in 0..27 {
        if i == 13 {
            bg[i] = true; // centre treated as background
        } else {
            bg[i] = !nbh[i];
        }
    }

    let mut visited = [false; 27];
    let mut count = 0_usize;

    // 6-connectivity offsets in flat index space (delta z*9 + delta y*3 + delta x)
    let face_deltas: [isize; 6] = [-9, 9, -3, 3, -1, 1];

    for start in 0..27_usize {
        if !bg[start] || visited[start] {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::<usize>::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(cur) = queue.pop_front() {
            let (cz, cy, cx) = ((cur / 9) as isize, ((cur % 9) / 3) as isize, (cur % 3) as isize);
            for &delta in &face_deltas {
                let ni_s = cur as isize + delta;
                if ni_s < 0 || ni_s >= 27 {
                    continue;
                }
                let ni = ni_s as usize;
                // Validate that the neighbour is actually adjacent (no wrap)
                let (nz, ny, nx) = ((ni / 9) as isize, ((ni % 9) / 3) as isize, (ni % 3) as isize);
                let d_abs = (nz - cz).abs() + (ny - cy).abs() + (nx - cx).abs();
                if d_abs != 1 {
                    continue;
                }
                if bg[ni] && !visited[ni] {
                    visited[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn filled_cube(sz: usize) -> Array3<bool> {
        Array3::from_elem((sz, sz, sz), true)
    }

    #[test]
    fn binary_erosion_shrinks() {
        let vol = filled_cube(7);
        let se = StructElem3D::Cross26;
        let out = binary_erosion_3d(vol.view(), &se).expect("binary erosion failed");
        // Interior voxels survive; border voxels are eroded away
        assert!(!out[[0, 0, 0]], "corner should be eroded");
        assert!(out[[3, 3, 3]], "center should survive");
    }

    #[test]
    fn binary_dilation_of_background_stays_background() {
        let vol = Array3::<bool>::from_elem((6, 6, 6), false);
        let se = StructElem3D::Cross26;
        let out = binary_dilation_3d(vol.view(), &se).expect("binary dilation failed");
        assert!(out.iter().all(|&v| !v));
    }

    #[test]
    fn opening_idempotent_on_full_cube() {
        let vol = Array3::from_elem((8, 8, 8), 5.0_f64);
        let se = StructElem3D::Cube(1);
        let opened = opening_3d(vol.view(), &se).expect("opening failed");
        // Opening a constant volume must return the same constant
        for v in opened.iter() {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn closing_idempotent_on_full_cube() {
        let vol = Array3::from_elem((8, 8, 8), 2.0_f64);
        let se = StructElem3D::Cube(1);
        let closed = closing_3d(vol.view(), &se).expect("closing failed");
        for v in closed.iter() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn connected_components_single_blob() {
        let mut vol = Array3::<bool>::from_elem((10, 10, 10), false);
        // Fill a 3×3×3 blob
        for z in 3..6 {
            for y in 3..6 {
                for x in 3..6 {
                    vol[[z, y, x]] = true;
                }
            }
        }
        let (labels, n) = connected_components_3d(vol.view()).expect("cc3d failed");
        assert_eq!(n, 1, "Expected 1 component, got {n}");
        assert_eq!(labels[[4, 4, 4]], 1);
        assert_eq!(labels[[0, 0, 0]], 0); // background
    }

    #[test]
    fn connected_components_two_blobs() {
        let mut vol = Array3::<bool>::from_elem((15, 15, 15), false);
        // Blob 1
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    vol[[z, y, x]] = true;
                }
            }
        }
        // Blob 2 — separated from blob 1
        for z in 10..13 {
            for y in 10..13 {
                for x in 10..13 {
                    vol[[z, y, x]] = true;
                }
            }
        }
        let (_labels, n) = connected_components_3d(vol.view()).expect("cc3d failed");
        assert_eq!(n, 2, "Expected 2 components, got {n}");
    }

    #[test]
    fn connected_components_rejects_empty() {
        let vol = Array3::<bool>::from_elem((0, 4, 4), false);
        assert!(connected_components_3d(vol.view()).is_err());
    }

    #[test]
    fn skeleton_single_voxel() {
        let mut vol = Array3::<bool>::from_elem((5, 5, 5), false);
        vol[[2, 2, 2]] = true;
        let skel = skeleton_3d(vol.view()).expect("skeleton failed");
        assert!(skel[[2, 2, 2]], "Single voxel must be preserved");
    }

    #[test]
    fn skeleton_reduces_ball() {
        let mut vol = Array3::<bool>::from_elem((9, 9, 9), false);
        // Fill a sphere of radius 3 centred at (4,4,4)
        for z in 0..9_usize {
            for y in 0..9_usize {
                for x in 0..9_usize {
                    let dz = z as f64 - 4.0;
                    let dy = y as f64 - 4.0;
                    let dx = x as f64 - 4.0;
                    if (dz * dz + dy * dy + dx * dx).sqrt() <= 3.5 {
                        vol[[z, y, x]] = true;
                    }
                }
            }
        }
        let fg_before = vol.iter().filter(|&&v| v).count();
        let skel = skeleton_3d(vol.view()).expect("skeleton failed");
        let fg_after = skel.iter().filter(|&&v| v).count();
        assert!(
            fg_after < fg_before,
            "Skeleton should reduce voxel count: before={fg_before} after={fg_after}"
        );
    }
}
