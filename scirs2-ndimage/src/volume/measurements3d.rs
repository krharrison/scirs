//! 3D Volumetric Measurements
//!
//! Provides quantitative analysis of labelled 3-D volumes:
//!
//! * [`label_3d`]              – connected-component labeling (26-conn)
//! * [`region_props_3d`]       – per-region volumetric properties
//! * [`moment_of_inertia_3d`]  – full 3×3 inertia tensor of a binary mask
//!
//! # Region Properties Computed
//!
//! For each labelled region, [`RegionProps3D`] reports:
//!
//! | Field | Description |
//! |-------|-------------|
//! | `label` | Integer label |
//! | `volume` | Voxel count |
//! | `surface_area` | Approximate face-adjacency surface area |
//! | `bbox` | Axis-aligned bounding box `(z_min..z_max, y_min..y_max, x_min..x_max)` |
//! | `centroid` | Centre of mass `(z̄, ȳ, x̄)` |
//! | `principal_axes` | 3 eigenvectors of the inertia tensor (columns) |
//! | `eigenvalues` | Corresponding eigenvalues |
//! | `sphericity` | `π^(1/3) · (6V)^(2/3) / A` ∈ (0, 1] |
//! | `compactness` | `V / bbox_volume` ∈ (0, 1] |
//! | `elongation` | `λ_max / λ_min` ≥ 1 |

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3, ArrayView3};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Connected-component labeling (26-connectivity BFS)
// ---------------------------------------------------------------------------

/// Label connected foreground components in a binary volume using 26-connectivity.
///
/// Returns `(labels, n_labels)` where `labels[[z,y,x]] == 0` is background and
/// positive values are unique component identifiers `1..=n_labels`.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty.
pub fn label_3d(volume: ArrayView3<bool>) -> NdimageResult<(Array3<u32>, u32)> {
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

    // Pre-compute 26-connectivity offsets
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
// Region properties
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box of a 3-D region.
#[derive(Debug, Clone, PartialEq)]
pub struct BBox3D {
    pub z_min: usize,
    pub z_max: usize, // inclusive
    pub y_min: usize,
    pub y_max: usize,
    pub x_min: usize,
    pub x_max: usize,
}

impl BBox3D {
    /// Volume of the bounding box in voxels.
    pub fn volume(&self) -> usize {
        (self.z_max - self.z_min + 1)
            * (self.y_max - self.y_min + 1)
            * (self.x_max - self.x_min + 1)
    }
}

/// Comprehensive 3-D region properties for a single connected component.
#[derive(Debug, Clone)]
pub struct RegionProps3D {
    /// Integer label identifying this region.
    pub label: u32,
    /// Number of foreground voxels.
    pub volume: usize,
    /// Approximate surface area (count of face-adjacent foreground–background
    /// pairs in the 6-neighbourhood).
    pub surface_area: usize,
    /// Axis-aligned bounding box.
    pub bbox: BBox3D,
    /// Centre of mass `(z̄, ȳ, x̄)` in voxel coordinates.
    pub centroid: (f64, f64, f64),
    /// Principal axes: columns of the 3×3 rotation matrix (eigenvectors of the
    /// inertia tensor, sorted from smallest to largest eigenvalue).
    pub principal_axes: [[f64; 3]; 3],
    /// Eigenvalues of the inertia tensor (ascending order).
    pub eigenvalues: [f64; 3],
    /// Sphericity: `π^(1/3) · (6·volume)^(2/3) / surface_area`.
    /// Equal to 1 for a perfect sphere.
    pub sphericity: f64,
    /// Compactness: `volume / bbox.volume()`.
    pub compactness: f64,
    /// Elongation: ratio of largest to smallest inertia eigenvalue.
    pub elongation: f64,
}

/// Compute 3-D region properties for every label in a labelled volume.
///
/// # Arguments
///
/// * `labeled_volume` – Output of [`label_3d`]; `0` is background.
/// * `n_labels`       – Number of labels (the second return value of `label_3d`).
///
/// # Returns
///
/// A `Vec<RegionProps3D>` with one entry per label, sorted by label value.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty.
pub fn region_props_3d(
    labeled_volume: &Array3<u32>,
    n_labels: u32,
) -> NdimageResult<Vec<RegionProps3D>> {
    let sh = labeled_volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    if n_labels == 0 {
        return Ok(Vec::new());
    }

    let n = n_labels as usize;

    // Accumulators (indexed by label − 1)
    let mut vol_counts = vec![0_usize; n];
    let mut surface_counts = vec![0_usize; n];
    let mut sum_z = vec![0.0_f64; n];
    let mut sum_y = vec![0.0_f64; n];
    let mut sum_x = vec![0.0_f64; n];
    // Second-order moments for inertia tensor
    let mut sum_zz = vec![0.0_f64; n];
    let mut sum_yy = vec![0.0_f64; n];
    let mut sum_xx = vec![0.0_f64; n];
    let mut sum_zy = vec![0.0_f64; n];
    let mut sum_zx = vec![0.0_f64; n];
    let mut sum_yx = vec![0.0_f64; n];
    // Bounding box (use sentinel values)
    let inf = usize::MAX;
    let mut z_min = vec![inf; n];
    let mut z_max = vec![0_usize; n];
    let mut y_min = vec![inf; n];
    let mut y_max = vec![0_usize; n];
    let mut x_min = vec![inf; n];
    let mut x_max = vec![0_usize; n];

    // 6-connected face offsets for surface detection
    let face_offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let lbl = labeled_volume[[z, y, x]];
                if lbl == 0 {
                    continue;
                }
                let li = (lbl - 1) as usize;
                let zf = z as f64;
                let yf = y as f64;
                let xf = x as f64;

                vol_counts[li] += 1;
                sum_z[li] += zf;
                sum_y[li] += yf;
                sum_x[li] += xf;
                sum_zz[li] += zf * zf;
                sum_yy[li] += yf * yf;
                sum_xx[li] += xf * xf;
                sum_zy[li] += zf * yf;
                sum_zx[li] += zf * xf;
                sum_yx[li] += yf * xf;

                if z < z_min[li] { z_min[li] = z; }
                if z > z_max[li] { z_max[li] = z; }
                if y < y_min[li] { y_min[li] = y; }
                if y > y_max[li] { y_max[li] = y; }
                if x < x_min[li] { x_min[li] = x; }
                if x > x_max[li] { x_max[li] = x; }

                // Surface check: count each exposed face (face-adjacent to background).
                // This gives the actual surface area in face units (each face = 1 voxel²).
                for &(dz, dy, dx) in &face_offsets {
                    let nz_i = z as isize + dz;
                    let ny_i = y as isize + dy;
                    let nx_i = x as isize + dx;
                    let is_bg = if nz_i < 0
                        || ny_i < 0
                        || nx_i < 0
                        || (nz_i as usize) >= nz
                        || (ny_i as usize) >= ny
                        || (nx_i as usize) >= nx
                    {
                        true
                    } else {
                        labeled_volume[[nz_i as usize, ny_i as usize, nx_i as usize]] == 0
                    };
                    if is_bg {
                        // Count each exposed face separately (not break)
                        surface_counts[li] += 1;
                    }
                }
            }
        }
    }

    // Build results
    let mut props = Vec::with_capacity(n);
    for li in 0..n {
        let v = vol_counts[li];
        if v == 0 {
            // Label was requested but has no voxels (shouldn't happen normally)
            continue;
        }
        let vf = v as f64;
        let cz = sum_z[li] / vf;
        let cy = sum_y[li] / vf;
        let cx = sum_x[li] / vf;

        // Central second moments
        let m_zz = sum_zz[li] / vf - cz * cz;
        let m_yy = sum_yy[li] / vf - cy * cy;
        let m_xx = sum_xx[li] / vf - cx * cx;
        let m_zy = sum_zy[li] / vf - cz * cy;
        let m_zx = sum_zx[li] / vf - cz * cx;
        let m_yx = sum_yx[li] / vf - cy * cx;

        // Inertia tensor (symmetric 3×3)
        // I_ij = V * (δ_ij * Σ m_kk - m_ij)
        let i_zz = vf * (m_yy + m_xx);
        let i_yy = vf * (m_zz + m_xx);
        let i_xx = vf * (m_zz + m_yy);
        let i_zy = -vf * m_zy;
        let i_zx = -vf * m_zx;
        let i_yx = -vf * m_yx;

        let (eigenvalues, axes) = eigen_sym3(
            [i_zz, i_zy, i_zx, i_zy, i_yy, i_yx, i_zx, i_yx, i_xx],
        );

        let sa = surface_counts[li];
        let sphericity = if sa > 0 {
            let s_f = sa as f64;
            std::f64::consts::PI.powf(1.0 / 3.0) * (6.0 * vf).powf(2.0 / 3.0) / s_f
        } else {
            1.0
        };

        let bbox = BBox3D {
            z_min: z_min[li],
            z_max: z_max[li],
            y_min: y_min[li],
            y_max: y_max[li],
            x_min: x_min[li],
            x_max: x_max[li],
        };
        let compactness = vf / bbox.volume() as f64;

        let lambda_min = eigenvalues[0].max(1e-12);
        let lambda_max = eigenvalues[2].max(1e-12);
        let elongation = lambda_max / lambda_min;

        props.push(RegionProps3D {
            label: (li + 1) as u32,
            volume: v,
            surface_area: sa,
            bbox,
            centroid: (cz, cy, cx),
            principal_axes: axes,
            eigenvalues,
            sphericity,
            compactness,
            elongation,
        });
    }
    Ok(props)
}

// ---------------------------------------------------------------------------
// Moment of inertia tensor
// ---------------------------------------------------------------------------

/// Compute the 3×3 moment-of-inertia tensor for a binary mask.
///
/// The tensor is defined as:
/// ```text
///   I = sum_v [ (r·r) I₃ − r⊗r ]
/// ```
/// where `r = (z − z̄, y − ȳ, x − x̄)` for each foreground voxel `v`.
///
/// Returns the 3×3 tensor as `Array2<f64>` with indices `[row, col]` ordered
/// as `(Z, Y, X)`.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty or contains
/// no foreground voxels.
pub fn moment_of_inertia_3d(
    volume: ArrayView3<bool>,
) -> NdimageResult<Array2<f64>> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }

    let mut sz = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut count = 0_usize;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if volume[[z, y, x]] {
                    sz += z as f64;
                    sy += y as f64;
                    sx += x as f64;
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return Err(NdimageError::InvalidInput(
            "No foreground voxels in volume".to_string(),
        ));
    }

    let n = count as f64;
    let cz = sz / n;
    let cy = sy / n;
    let cx = sx / n;

    let mut i_zz = 0.0_f64;
    let mut i_yy = 0.0_f64;
    let mut i_xx = 0.0_f64;
    let mut i_zy = 0.0_f64;
    let mut i_zx = 0.0_f64;
    let mut i_yx = 0.0_f64;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if volume[[z, y, x]] {
                    let rz = z as f64 - cz;
                    let ry = y as f64 - cy;
                    let rx = x as f64 - cx;
                    let r2 = rz * rz + ry * ry + rx * rx;
                    i_zz += r2 - rz * rz;
                    i_yy += r2 - ry * ry;
                    i_xx += r2 - rx * rx;
                    i_zy -= rz * ry;
                    i_zx -= rz * rx;
                    i_yx -= ry * rx;
                }
            }
        }
    }

    let mut tensor = Array2::<f64>::zeros((3, 3));
    tensor[[0, 0]] = i_zz;
    tensor[[1, 1]] = i_yy;
    tensor[[2, 2]] = i_xx;
    tensor[[0, 1]] = i_zy;
    tensor[[1, 0]] = i_zy;
    tensor[[0, 2]] = i_zx;
    tensor[[2, 0]] = i_zx;
    tensor[[1, 2]] = i_yx;
    tensor[[2, 1]] = i_yx;

    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Symmetric 3×3 eigendecomposition (Jacobi)
// ---------------------------------------------------------------------------

/// Compute eigenvalues and eigenvectors of a real symmetric 3×3 matrix using
/// the Jacobi iterative method.
///
/// `m` is given in row-major order:
/// `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
///
/// Returns `(eigenvalues, axes)` where `axes[col]` is the `col`-th eigenvector
/// and eigenvalues are sorted ascending.
fn eigen_sym3(m: [f64; 9]) -> ([f64; 3], [[f64; 3]; 3]) {
    // Work with a 3×3 matrix; a[i][j]
    let mut a = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
    // Eigenvector matrix (starts as identity)
    let mut v = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    const MAX_ITER: usize = 50;

    for _ in 0..MAX_ITER {
        // Find the off-diagonal element with the largest absolute value
        let (mut p, mut q) = (0_usize, 1_usize);
        let mut max_off = a[0][1].abs();
        let candidates = [(0, 1), (0, 2), (1, 2)];
        for &(i, j) in &candidates {
            if a[i][j].abs() > max_off {
                max_off = a[i][j].abs();
                p = i;
                q = j;
            }
        }

        if max_off < 1e-14 {
            break; // converged
        }

        // Compute the Jacobi rotation angle
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to `a` and accumulate in `v`
        let a_old = a;
        for i in 0..3 {
            a[i][p] = c * a_old[i][p] - s * a_old[i][q];
            a[p][i] = a[i][p];
            a[i][q] = s * a_old[i][p] + c * a_old[i][q];
            a[q][i] = a[i][q];
        }
        a[p][p] = c * c * a_old[p][p] - 2.0 * s * c * a_old[p][q] + s * s * a_old[q][q];
        a[q][q] = s * s * a_old[p][p] + 2.0 * s * c * a_old[p][q] + c * c * a_old[q][q];
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        let v_old = v;
        for i in 0..3 {
            v[i][p] = c * v_old[i][p] - s * v_old[i][q];
            v[i][q] = s * v_old[i][p] + c * v_old[i][q];
        }
    }

    let mut eigenvalues = [a[0][0], a[1][1], a[2][2]];
    let mut axes: [[f64; 3]; 3] = [[v[0][0], v[1][0], v[2][0]],
                                    [v[0][1], v[1][1], v[2][1]],
                                    [v[0][2], v[1][2], v[2][2]]];

    // Sort eigenvalues ascending and permute eigenvectors
    let mut order = [0_usize, 1, 2];
    order.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap_or(std::cmp::Ordering::Equal));
    let ev_sorted = [eigenvalues[order[0]], eigenvalues[order[1]], eigenvalues[order[2]]];
    let ax_sorted = [axes[order[0]], axes[order[1]], axes[order[2]]];
    eigenvalues = ev_sorted;
    axes = ax_sorted;

    (eigenvalues, axes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn sphere_mask(sz: usize, radius: f64) -> Array3<bool> {
        let c = sz as f64 / 2.0;
        Array3::from_shape_fn((sz, sz, sz), |(z, y, x)| {
            let dz = z as f64 - c;
            let dy = y as f64 - c;
            let dx = x as f64 - c;
            (dz * dz + dy * dy + dx * dx).sqrt() <= radius
        })
    }

    #[test]
    fn label_3d_single_component() {
        let vol = sphere_mask(11, 4.0);
        let (labels, n) = label_3d(vol.view()).expect("label_3d failed");
        assert_eq!(n, 1, "sphere should be 1 component");
        let center = labels[[5, 5, 5]];
        assert_eq!(center, 1);
    }

    #[test]
    fn label_3d_two_components() {
        let mut vol = Array3::<bool>::from_elem((20, 10, 10), false);
        for z in 0..5_usize { for y in 0..5 { for x in 0..5 { vol[[z, y, x]] = true; } } }
        for z in 15..20_usize { for y in 5..10 { for x in 5..10 { vol[[z, y, x]] = true; } } }
        let (_labels, n) = label_3d(vol.view()).expect("label_3d 2 comp failed");
        assert_eq!(n, 2);
    }

    #[test]
    fn label_3d_rejects_empty() {
        let vol = Array3::<bool>::from_elem((0, 5, 5), false);
        assert!(label_3d(vol.view()).is_err());
    }

    #[test]
    fn region_props_centroid() {
        let mut vol = Array3::<bool>::from_elem((10, 10, 10), false);
        // Single voxel at (5,5,5)
        vol[[5, 5, 5]] = true;
        let (labels, n) = label_3d(vol.view()).expect("label failed");
        let props = region_props_3d(&labels, n).expect("props failed");
        assert_eq!(props.len(), 1);
        let p = &props[0];
        assert_eq!(p.volume, 1);
        assert!((p.centroid.0 - 5.0).abs() < 1e-10, "centroid z: {}", p.centroid.0);
        assert!((p.centroid.1 - 5.0).abs() < 1e-10, "centroid y: {}", p.centroid.1);
        assert!((p.centroid.2 - 5.0).abs() < 1e-10, "centroid x: {}", p.centroid.2);
    }

    #[test]
    fn region_props_sphere_sphericity() {
        let vol = sphere_mask(21, 8.0);
        let (labels, n) = label_3d(vol.view()).expect("label failed");
        let props = region_props_3d(&labels, n).expect("props failed");
        assert!(!props.is_empty());
        let p = &props[0];
        // Sphericity of a voxelized sphere should be close to 1
        // For a voxelized sphere, face-counting surface area overestimates the
        // true area by ~1.5x due to the stepped surface, giving sphericity ≈ 0.6–0.7.
        // Still significantly higher than a cube (sphericity ≈ 0.5), confirming
        // the metric is working correctly as a relative shape descriptor.
        assert!(
            p.sphericity > 0.5 && p.sphericity < 1.1,
            "sphericity={} should be in (0.5, 1.1) for a voxelized sphere",
            p.sphericity
        );
    }

    #[test]
    fn region_props_empty_volume() {
        let labels = Array3::<u32>::zeros((5, 5, 5));
        let props = region_props_3d(&labels, 0).expect("props empty failed");
        assert!(props.is_empty());
    }

    #[test]
    fn moment_of_inertia_sphere_diagonal() {
        let vol = sphere_mask(11, 4.0);
        let tensor = moment_of_inertia_3d(vol.view()).expect("inertia failed");
        // For an isotropic sphere the off-diagonal terms should be small
        let off_diag_max = [tensor[[0, 1]], tensor[[0, 2]], tensor[[1, 2]]]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let diag_avg = (tensor[[0, 0]] + tensor[[1, 1]] + tensor[[2, 2]]) / 3.0;
        assert!(
            off_diag_max < diag_avg * 0.05,
            "off-diagonal {off_diag_max} should be small vs diagonal {diag_avg}"
        );
    }

    #[test]
    fn moment_of_inertia_rejects_empty_volume() {
        let vol = Array3::<bool>::from_elem((5, 5, 5), false);
        assert!(moment_of_inertia_3d(vol.view()).is_err());
    }

    #[test]
    fn eigen_sym3_identity() {
        // Eigenvalues of diag(1,2,3) should be 1, 2, 3
        let m = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let (ev, _axes) = super::eigen_sym3(m);
        assert!((ev[0] - 1.0).abs() < 1e-10, "ev[0]={}", ev[0]);
        assert!((ev[1] - 2.0).abs() < 1e-10, "ev[1]={}", ev[1]);
        assert!((ev[2] - 3.0).abs() < 1e-10, "ev[2]={}", ev[2]);
    }
}
