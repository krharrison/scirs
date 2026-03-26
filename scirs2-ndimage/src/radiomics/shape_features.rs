//! 3D shape-based radiomics features
//!
//! Computes PyRadiomics-compatible 3D shape features from a binary mask.

use std::f64::consts::PI;

/// 3D shape-based radiomics features computed from a binary segmentation mask.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeFeatures {
    /// Number of True voxels
    pub volume: f64,
    /// Surface area estimated from exposed voxel faces (marching-cubes approximation)
    pub surface_area: f64,
    /// surface_area / volume
    pub surface_volume_ratio: f64,
    /// Compactness1 = V / (pi^(1/3) * (6V)^(2/3))
    pub compactness1: f64,
    /// Compactness2 = 36*pi * V^2 / SA^3
    pub compactness2: f64,
    /// Sphericity = (pi^(1/3) * (6V)^(2/3)) / SA
    pub sphericity: f64,
    /// Asphericity = (SA^3 / (36*pi * V^2))^(1/3) - 1
    pub asphericity: f64,
    /// Elongation = sqrt(lambda2 / lambda1) from PCA
    pub elongation: f64,
    /// Flatness = sqrt(lambda3 / lambda1) from PCA
    pub flatness: f64,
    /// Approximate maximum 3D diameter from sampled surface voxels
    pub maximum_diameter: f64,
    /// Square root of the largest PCA eigenvalue
    pub pca_major_axis: f64,
    /// Square root of the second PCA eigenvalue
    pub pca_minor_axis: f64,
    /// Square root of the smallest PCA eigenvalue
    pub pca_least_axis: f64,
}

/// Compute 3D shape-based radiomics features from a binary 3D mask.
///
/// The mask is a `z × y × x` array where `true` indicates foreground voxels.
/// All voxels are assumed to be unit cubes (1×1×1 mm).
///
/// Returns `ShapeFeatures` with all metrics. If the mask is empty, most
/// features default to 0 or `f64::NAN`.
pub fn compute_shape_features(mask: &[Vec<Vec<bool>>]) -> ShapeFeatures {
    let nz = mask.len();
    if nz == 0 {
        return zero_features();
    }
    let ny = mask[0].len();
    let nx = if ny > 0 { mask[0][0].len() } else { 0 };

    // --- Volume ---
    let mut volume: usize = 0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if mask[z][y][x] {
                    volume += 1;
                }
            }
        }
    }

    if volume == 0 {
        return zero_features();
    }
    let vol_f = volume as f64;

    // --- Surface area via exposed-face counting ---
    // Each face that borders either a background voxel or the image boundary
    // contributes 1 unit of surface area (since voxel size = 1.0).
    let mut sa_faces: usize = 0;
    let nz_i = nz as isize;
    let ny_i = ny as isize;
    let nx_i = nx as isize;

    // Helper: is voxel at (iz,iy,ix) in foreground?
    let in_mask = |iz: isize, iy: isize, ix: isize| -> bool {
        if iz < 0 || iz >= nz_i || iy < 0 || iy >= ny_i || ix < 0 || ix >= nx_i {
            return false;
        }
        mask[iz as usize][iy as usize][ix as usize]
    };

    // Collect surface voxels (for max diameter) while counting faces
    let mut surface_coords: Vec<[f64; 3]> = Vec::new();

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if !mask[z][y][x] {
                    continue;
                }
                let iz = z as isize;
                let iy = y as isize;
                let ix = x as isize;

                let mut local_faces = 0usize;
                for (dz, dy, dx) in &[
                    (-1_isize, 0_isize, 0_isize),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    if !in_mask(iz + dz, iy + dy, ix + dx) {
                        local_faces += 1;
                    }
                }
                if local_faces > 0 {
                    sa_faces += local_faces;
                    surface_coords.push([z as f64, y as f64, x as f64]);
                }
            }
        }
    }

    // Surface area = number of exposed faces × face area (=1 for unit voxels)
    let surface_area = sa_faces as f64;

    // --- Shape ratios ---
    let surface_volume_ratio = if vol_f > 0.0 {
        surface_area / vol_f
    } else {
        f64::NAN
    };

    // Compactness1 = V / (pi^(1/3) * (6V)^(2/3))
    let compactness1 = if vol_f > 0.0 {
        let denom = PI.powf(1.0 / 3.0) * (6.0 * vol_f).powf(2.0 / 3.0);
        if denom > 0.0 {
            vol_f / denom
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    // Compactness2 = 36*pi * V^2 / SA^3
    let compactness2 = if surface_area > 0.0 {
        36.0 * PI * vol_f * vol_f / (surface_area * surface_area * surface_area)
    } else {
        f64::NAN
    };

    // Sphericity = (pi^(1/3) * (6V)^(2/3)) / SA
    let sphericity = if surface_area > 0.0 {
        PI.powf(1.0 / 3.0) * (6.0 * vol_f).powf(2.0 / 3.0) / surface_area
    } else {
        f64::NAN
    };

    // Asphericity = (SA^3 / (36*pi * V^2))^(1/3) - 1
    let asphericity = if vol_f > 0.0 {
        let ratio = (surface_area * surface_area * surface_area) / (36.0 * PI * vol_f * vol_f);
        ratio.powf(1.0 / 3.0) - 1.0
    } else {
        f64::NAN
    };

    // --- PCA on foreground voxel coordinates ---
    // Collect all foreground voxel centres
    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(volume);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if mask[z][y][x] {
                    coords.push([z as f64, y as f64, x as f64]);
                }
            }
        }
    }

    let (lambda1, lambda2, lambda3) = pca_eigenvalues_3x3(&coords);

    let pca_major_axis = lambda1.max(0.0).sqrt();
    let pca_minor_axis = lambda2.max(0.0).sqrt();
    let pca_least_axis = lambda3.max(0.0).sqrt();

    let elongation = if lambda1 > 0.0 {
        (lambda2 / lambda1).max(0.0).sqrt()
    } else {
        f64::NAN
    };
    let flatness = if lambda1 > 0.0 {
        (lambda3 / lambda1).max(0.0).sqrt()
    } else {
        f64::NAN
    };

    // --- Maximum 3D diameter ---
    // Compute from sampled surface voxels to avoid O(N^2) cost on large surfaces.
    let max_diameter = compute_max_diameter(&surface_coords);

    ShapeFeatures {
        volume: vol_f,
        surface_area,
        surface_volume_ratio,
        compactness1,
        compactness2,
        sphericity,
        asphericity,
        elongation,
        flatness,
        maximum_diameter: max_diameter,
        pca_major_axis,
        pca_minor_axis,
        pca_least_axis,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return a zeroed / NaN feature set for empty masks.
fn zero_features() -> ShapeFeatures {
    ShapeFeatures {
        volume: 0.0,
        surface_area: 0.0,
        surface_volume_ratio: f64::NAN,
        compactness1: f64::NAN,
        compactness2: f64::NAN,
        sphericity: f64::NAN,
        asphericity: f64::NAN,
        elongation: f64::NAN,
        flatness: f64::NAN,
        maximum_diameter: 0.0,
        pca_major_axis: 0.0,
        pca_minor_axis: 0.0,
        pca_least_axis: 0.0,
    }
}

/// Compute the three eigenvalues of the covariance matrix of 3-D points,
/// returned in descending order (λ1 ≥ λ2 ≥ λ3).
///
/// Uses power iteration on the symmetric 3×3 covariance matrix with two
/// deflation steps so that no external linear algebra crate is needed.
fn pca_eigenvalues_3x3(coords: &[[f64; 3]]) -> (f64, f64, f64) {
    let n = coords.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }

    // Compute centroid
    let mut mean = [0.0f64; 3];
    for c in coords {
        mean[0] += c[0];
        mean[1] += c[1];
        mean[2] += c[2];
    }
    let nf = n as f64;
    mean[0] /= nf;
    mean[1] /= nf;
    mean[2] /= nf;

    // Build 3×3 covariance matrix (symmetric)
    let mut cov = [[0.0f64; 3]; 3];
    for c in coords {
        let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= nf;
        }
    }

    // Power iteration + deflation to get 3 eigenvalues
    let (lam1, v1) = power_iter_3x3(&cov, 200);
    // Deflate: cov2 = cov - lam1 * v1 ⊗ v1
    let mut cov2 = cov;
    for i in 0..3 {
        for j in 0..3 {
            cov2[i][j] -= lam1 * v1[i] * v1[j];
        }
    }
    let (lam2, v2) = power_iter_3x3(&cov2, 200);
    // Second deflation
    let mut cov3 = cov2;
    for i in 0..3 {
        for j in 0..3 {
            cov3[i][j] -= lam2 * v2[i] * v2[j];
        }
    }
    let (lam3, _) = power_iter_3x3(&cov3, 200);

    // Sort descending
    let mut eigs = [lam1, lam2, lam3];
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    (eigs[0], eigs[1], eigs[2])
}

/// Power iteration for a symmetric 3×3 matrix.
/// Returns (dominant eigenvalue, corresponding unit eigenvector).
fn power_iter_3x3(a: &[[f64; 3]; 3], max_iter: usize) -> (f64, [f64; 3]) {
    // Start with a non-trivial vector
    let mut v = [1.0f64, 0.5, 0.25];
    normalize_3(&mut v);

    let mut eigenvalue = 0.0f64;
    for _ in 0..max_iter {
        let mut av = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                av[i] += a[i][j] * v[j];
            }
        }
        let norm = vec3_norm(&av);
        if norm < 1e-15 {
            break;
        }
        eigenvalue = norm;
        // Sign: ensure av · v > 0 so the eigenvector direction is stable
        let dot: f64 = av.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            eigenvalue = -norm;
        }
        v = [av[0] / norm, av[1] / norm, av[2] / norm];
    }
    (eigenvalue, v)
}

fn vec3_norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize_3(v: &mut [f64; 3]) {
    let n = vec3_norm(v);
    if n > 1e-15 {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

/// Compute approximate maximum pairwise Euclidean distance among surface voxels.
/// For large sets we subsample uniformly to keep cost manageable.
fn compute_max_diameter(surface_coords: &[[f64; 3]]) -> f64 {
    let n = surface_coords.len();
    if n == 0 {
        return 0.0;
    }
    // Subsample: use at most 512 representative points
    let step = if n > 512 { n / 512 } else { 1 };
    let sampled: Vec<&[f64; 3]> = surface_coords.iter().step_by(step).collect();
    let m = sampled.len();

    let mut max_sq = 0.0f64;
    for i in 0..m {
        for j in (i + 1)..m {
            let dz = sampled[i][0] - sampled[j][0];
            let dy = sampled[i][1] - sampled[j][1];
            let dx = sampled[i][2] - sampled[j][2];
            let sq = dz * dz + dy * dy + dx * dx;
            if sq > max_sq {
                max_sq = sq;
            }
        }
    }
    max_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3×3×3 cube mask (all true).
    fn cube_mask_3x3x3() -> Vec<Vec<Vec<bool>>> {
        vec![vec![vec![true; 3]; 3]; 3]
    }

    #[test]
    fn test_cube_volume() {
        let mask = cube_mask_3x3x3();
        let f = compute_shape_features(&mask);
        assert_eq!(f.volume, 27.0, "volume of 3×3×3 cube should be 27");
    }

    #[test]
    fn test_cube_surface_area() {
        let mask = cube_mask_3x3x3();
        let f = compute_shape_features(&mask);
        // 3×3×3 cube: 6 faces × 9 voxels per face = 54 exposed face-units
        assert_eq!(f.surface_area, 54.0);
    }

    #[test]
    fn test_cube_sphericity_range() {
        let mask = cube_mask_3x3x3();
        let f = compute_shape_features(&mask);
        // Sphericity must be in (0, 1] — a cube is less spherical than a sphere
        assert!(
            f.sphericity > 0.0 && f.sphericity <= 1.0,
            "sphericity = {} should be in (0, 1]",
            f.sphericity
        );
    }

    #[test]
    fn test_cube_compactness2_positive() {
        let mask = cube_mask_3x3x3();
        let f = compute_shape_features(&mask);
        assert!(f.compactness2 > 0.0);
    }

    #[test]
    fn test_single_voxel() {
        let mask = vec![vec![vec![true]]];
        let f = compute_shape_features(&mask);
        assert_eq!(f.volume, 1.0);
        assert_eq!(f.surface_area, 6.0); // all 6 faces exposed
    }

    #[test]
    fn test_empty_mask() {
        let mask: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; 3]; 3]; 3];
        let f = compute_shape_features(&mask);
        assert_eq!(f.volume, 0.0);
        assert!(f.sphericity.is_nan());
    }

    #[test]
    fn test_asphericity_positive_for_cube() {
        let mask = cube_mask_3x3x3();
        let f = compute_shape_features(&mask);
        // For any non-sphere the asphericity must be > 0
        assert!(f.asphericity > 0.0, "asphericity = {}", f.asphericity);
    }
}
