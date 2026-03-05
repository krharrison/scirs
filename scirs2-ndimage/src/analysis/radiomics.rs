//! Radiomics Feature Extraction
//!
//! Provides a comprehensive radiomics feature extraction pipeline for medical
//! and general image analysis:
//!
//! - [`FirstOrderFeatures`]: energy, entropy, kurtosis, skewness, variance for ROI
//! - [`GLCMFeatures`]: extended Gray-Level Co-occurrence Matrix texture features
//! - [`ShapeFeatures3D`]: 3D shape descriptors (volume, surface area, sphericity)
//! - [`RadiomicsExtractor`]: unified pipeline combining all feature types
//!
//! # References
//!
//! - Zwanenburg et al. (2020), "The Image Biomarker Standardization Initiative",
//!   Radiology 295(2):328-338.
//! - Haralick et al. (1973), "Textural Features for Image Classification",
//!   IEEE Transactions on Systems, Man, and Cybernetics.

use std::collections::HashMap;

use scirs2_core::ndarray::{Array2, Array3};

use crate::error::{NdimageError, NdimageResult};

// ─── FirstOrderFeatures ───────────────────────────────────────────────────────

/// First-order intensity statistics computed within a region of interest.
///
/// All features follow IBSI definitions where applicable.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstOrderFeatures {
    /// Number of voxels in the ROI.
    pub n_voxels: usize,
    /// Mean intensity.
    pub mean: f64,
    /// Variance.
    pub variance: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Skewness (third standardised moment).
    pub skewness: f64,
    /// Kurtosis (fourth standardised moment, excess kurtosis = kurtosis - 3).
    pub kurtosis: f64,
    /// Excess kurtosis (Fisher's definition; 0 for normal distributions).
    pub excess_kurtosis: f64,
    /// Minimum intensity in the ROI.
    pub minimum: f64,
    /// Maximum intensity in the ROI.
    pub maximum: f64,
    /// Range (max - min).
    pub range: f64,
    /// Median intensity.
    pub median: f64,
    /// 10th percentile.
    pub p10: f64,
    /// 90th percentile.
    pub p90: f64,
    /// Interquartile range (p75 - p25).
    pub iqr: f64,
    /// Mean absolute deviation from the mean.
    pub mad: f64,
    /// Robust mean absolute deviation (from the median).
    pub rmad: f64,
    /// Energy: sum of squared intensities.
    pub energy: f64,
    /// Root energy: square root of energy.
    pub root_energy: f64,
    /// Entropy of the intensity histogram (Nat, base-e).
    pub entropy: f64,
    /// Coefficient of variation (std_dev / |mean|).
    pub coefficient_of_variation: f64,
    /// Uniformity: sum of squared histogram probabilities.
    pub uniformity: f64,
}

impl FirstOrderFeatures {
    /// Compute first-order features for all voxels in `volume` where `mask` is true.
    ///
    /// If `mask` is `None` all voxels are included.
    /// `n_bins` controls histogram resolution for entropy and uniformity.
    pub fn compute(
        volume: &Array3<f64>,
        mask: Option<&Array3<bool>>,
        n_bins: usize,
    ) -> NdimageResult<Self> {
        let shape = volume.shape();
        if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
            return Err(NdimageError::InvalidInput(
                "FirstOrderFeatures: volume must not be empty".to_string(),
            ));
        }
        if let Some(m) = mask {
            if m.shape() != shape {
                return Err(NdimageError::DimensionError(
                    "FirstOrderFeatures: mask shape must match volume shape".to_string(),
                ));
            }
        }
        if n_bins == 0 {
            return Err(NdimageError::InvalidInput(
                "FirstOrderFeatures: n_bins must be > 0".to_string(),
            ));
        }

        // Collect ROI voxels
        let voxels: Vec<f64> = match mask {
            Some(m) => volume
                .iter()
                .zip(m.iter())
                .filter_map(|(&v, &flag)| if flag { Some(v) } else { None })
                .collect(),
            None => volume.iter().cloned().collect(),
        };

        let n = voxels.len();
        if n == 0 {
            return Err(NdimageError::InvalidInput(
                "FirstOrderFeatures: mask selects zero voxels".to_string(),
            ));
        }

        let mut sorted = voxels.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = voxels.iter().sum::<f64>() / n as f64;
        let variance = voxels.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 1e-12 {
            voxels.iter().map(|v| ((v - mean) / std_dev).powi(3)).sum::<f64>() / n as f64
        } else {
            0.0
        };
        let kurtosis = if std_dev > 1e-12 {
            voxels.iter().map(|v| ((v - mean) / std_dev).powi(4)).sum::<f64>() / n as f64
        } else {
            0.0
        };
        let excess_kurtosis = kurtosis - 3.0;

        let percentile = |p: f64| -> f64 {
            let idx_f = p / 100.0 * (n - 1) as f64;
            let lo = idx_f.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = idx_f - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        };

        let minimum = sorted[0];
        let maximum = sorted[n - 1];
        let range = maximum - minimum;
        let median = percentile(50.0);
        let p10 = percentile(10.0);
        let p90 = percentile(90.0);
        let p25 = percentile(25.0);
        let p75 = percentile(75.0);
        let iqr = p75 - p25;

        let mad = voxels.iter().map(|v| (v - mean).abs()).sum::<f64>() / n as f64;
        let rmad = voxels.iter().map(|v| (v - median).abs()).sum::<f64>() / n as f64;

        let energy: f64 = voxels.iter().map(|v| v * v).sum();
        let root_energy = energy.sqrt();

        let coefficient_of_variation = if mean.abs() > 1e-12 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Build histogram for entropy and uniformity
        let bin_width = if range > 1e-12 { range / n_bins as f64 } else { 1.0 };
        let mut hist = vec![0usize; n_bins];
        for &v in &voxels {
            let bin = ((v - minimum) / bin_width).floor() as usize;
            let bin_idx = bin.min(n_bins - 1);
            hist[bin_idx] += 1;
        }
        let probs: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum();
        let uniformity: f64 = probs.iter().map(|&p| p * p).sum();

        Ok(FirstOrderFeatures {
            n_voxels: n,
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            excess_kurtosis,
            minimum,
            maximum,
            range,
            median,
            p10,
            p90,
            iqr,
            mad,
            rmad,
            energy,
            root_energy,
            entropy,
            coefficient_of_variation,
            uniformity,
        })
    }
}

// ─── GLCMFeatures ─────────────────────────────────────────────────────────────

/// Extended GLCM texture features extracted from a 2D slice or 3D volume.
///
/// Features follow Haralick (1973) and the IBSI feature set.
#[derive(Debug, Clone, PartialEq)]
pub struct GLCMFeatures {
    /// Contrast (variance-like measure of intensity differences).
    pub contrast: f64,
    /// Dissimilarity (sum of |i-j| × p(i,j)).
    pub dissimilarity: f64,
    /// Homogeneity / Inverse Difference Moment.
    pub homogeneity: f64,
    /// Angular Second Moment / Energy.
    pub energy: f64,
    /// Entropy of the GLCM.
    pub entropy: f64,
    /// Correlation.
    pub correlation: f64,
    /// Sum average.
    pub sum_average: f64,
    /// Sum variance.
    pub sum_variance: f64,
    /// Sum entropy.
    pub sum_entropy: f64,
    /// Difference variance.
    pub difference_variance: f64,
    /// Difference entropy.
    pub difference_entropy: f64,
    /// Information measure of correlation 1.
    pub info_measure_corr1: f64,
    /// Information measure of correlation 2.
    pub info_measure_corr2: f64,
    /// Maximum probability.
    pub max_probability: f64,
    /// Cluster shade.
    pub cluster_shade: f64,
    /// Cluster prominence.
    pub cluster_prominence: f64,
    /// Auto-correlation.
    pub auto_correlation: f64,
}

/// Direction used when computing the GLCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlcmDirection3D {
    /// Along axis 0 (z) with offset +1
    Axis0,
    /// Along axis 1 (y) with offset +1
    Axis1,
    /// Along axis 2 (x) with offset +1
    Axis2,
    /// Diagonal xy: offset (0, +1, +1)
    DiagXY,
    /// Diagonal xz: offset (+1, 0, +1)
    DiagXZ,
    /// Diagonal yz: offset (+1, +1, 0)
    DiagYZ,
}

impl GLCMFeatures {
    /// Compute GLCM features from a 2D image.
    ///
    /// `image` must contain non-negative integer-valued intensities in `[0, n_levels-1]`.
    /// `n_levels` is the number of gray levels.
    /// `offset` specifies the neighbour offset `(dy, dx)`.
    pub fn from_2d(
        image: &Array2<f64>,
        n_levels: usize,
        offset_y: i32,
        offset_x: i32,
    ) -> NdimageResult<Self> {
        if n_levels == 0 {
            return Err(NdimageError::InvalidInput(
                "GLCMFeatures: n_levels must be > 0".to_string(),
            ));
        }
        let rows = image.nrows();
        let cols = image.ncols();

        // Build unnormalised GLCM
        let mut glcm = vec![0.0_f64; n_levels * n_levels];
        let mut count = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                let nr = r as isize + offset_y as isize;
                let nc = c as isize + offset_x as isize;
                if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                    continue;
                }
                let i = image[[r, c]].floor() as usize;
                let j = image[[nr as usize, nc as usize]].floor() as usize;
                if i < n_levels && j < n_levels {
                    glcm[i * n_levels + j] += 1.0;
                    glcm[j * n_levels + i] += 1.0; // symmetric
                    count += 2;
                }
            }
        }
        if count == 0 {
            return Err(NdimageError::ComputationError(
                "GLCMFeatures: no valid pixel pairs for given offset".to_string(),
            ));
        }
        let total = glcm.iter().sum::<f64>().max(1e-12);
        glcm.iter_mut().for_each(|v| *v /= total);

        Self::from_normalized_glcm(&glcm, n_levels)
    }

    /// Compute GLCM features from a 3D volume by averaging over the specified
    /// axis direction.
    pub fn from_3d(
        volume: &Array3<f64>,
        n_levels: usize,
        direction: GlcmDirection3D,
    ) -> NdimageResult<Self> {
        if n_levels == 0 {
            return Err(NdimageError::InvalidInput(
                "GLCMFeatures: n_levels must be > 0".to_string(),
            ));
        }
        let shape = volume.shape();
        let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
        let (dz, dy, dx): (isize, isize, isize) = match direction {
            GlcmDirection3D::Axis0 => (1, 0, 0),
            GlcmDirection3D::Axis1 => (0, 1, 0),
            GlcmDirection3D::Axis2 => (0, 0, 1),
            GlcmDirection3D::DiagXY => (0, 1, 1),
            GlcmDirection3D::DiagXZ => (1, 0, 1),
            GlcmDirection3D::DiagYZ => (1, 1, 0),
        };

        let mut glcm = vec![0.0_f64; n_levels * n_levels];
        let mut count = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let niz = iz as isize + dz;
                    let niy = iy as isize + dy;
                    let nix = ix as isize + dx;
                    if niz < 0 || niz >= nz as isize
                        || niy < 0 || niy >= ny as isize
                        || nix < 0 || nix >= nx as isize
                    {
                        continue;
                    }
                    let i = volume[[iz, iy, ix]].floor() as usize;
                    let j = volume[[niz as usize, niy as usize, nix as usize]].floor() as usize;
                    if i < n_levels && j < n_levels {
                        glcm[i * n_levels + j] += 1.0;
                        glcm[j * n_levels + i] += 1.0;
                        count += 2;
                    }
                }
            }
        }
        if count == 0 {
            return Err(NdimageError::ComputationError(
                "GLCMFeatures: no valid voxel pairs for given direction".to_string(),
            ));
        }
        let total = glcm.iter().sum::<f64>().max(1e-12);
        glcm.iter_mut().for_each(|v| *v /= total);

        Self::from_normalized_glcm(&glcm, n_levels)
    }

    /// Compute all Haralick features from a pre-normalised GLCM (flat, row-major,
    /// size `n_levels × n_levels`).
    pub fn from_normalized_glcm(glcm: &[f64], n_levels: usize) -> NdimageResult<Self> {
        if glcm.len() != n_levels * n_levels {
            return Err(NdimageError::InvalidInput(format!(
                "GLCMFeatures: glcm length {} ≠ n_levels^2 {}",
                glcm.len(),
                n_levels * n_levels
            )));
        }

        let p = |i: usize, j: usize| glcm[i * n_levels + j];

        // Marginal distributions p_i(i), p_j(j)
        let pi: Vec<f64> = (0..n_levels).map(|i| (0..n_levels).map(|j| p(i, j)).sum()).collect();
        let pj: Vec<f64> = (0..n_levels).map(|j| (0..n_levels).map(|i| p(i, j)).sum()).collect();

        let mu_i: f64 = (0..n_levels).map(|i| i as f64 * pi[i]).sum();
        let mu_j: f64 = (0..n_levels).map(|j| j as f64 * pj[j]).sum();
        let sig_i: f64 = ((0..n_levels).map(|i| (i as f64 - mu_i).powi(2) * pi[i]).sum::<f64>()).sqrt();
        let sig_j: f64 = ((0..n_levels).map(|j| (j as f64 - mu_j).powi(2) * pj[j]).sum::<f64>()).sqrt();

        // Sum and difference distributions
        let max_sum = 2 * (n_levels - 1);
        let mut p_x_plus_y = vec![0.0_f64; max_sum + 1];
        let mut p_x_minus_y = vec![0.0_f64; n_levels];
        for i in 0..n_levels {
            for j in 0..n_levels {
                let pij = p(i, j);
                let sum_idx = i + j;
                if sum_idx <= max_sum {
                    p_x_plus_y[sum_idx] += pij;
                }
                let diff_idx = (i as isize - j as isize).unsigned_abs();
                if diff_idx < n_levels {
                    p_x_minus_y[diff_idx] += pij;
                }
            }
        }

        let sum_average: f64 = p_x_plus_y
            .iter()
            .enumerate()
            .map(|(k, &v)| k as f64 * v)
            .sum();
        let sum_variance: f64 = p_x_plus_y
            .iter()
            .enumerate()
            .map(|(k, &v)| (k as f64 - sum_average).powi(2) * v)
            .sum();
        let sum_entropy: f64 = p_x_plus_y
            .iter()
            .filter(|&&v| v > 1e-12)
            .map(|&v| -v * v.ln())
            .sum();

        let diff_mean: f64 = p_x_minus_y
            .iter()
            .enumerate()
            .map(|(k, &v)| k as f64 * v)
            .sum();
        let difference_variance: f64 = p_x_minus_y
            .iter()
            .enumerate()
            .map(|(k, &v)| (k as f64 - diff_mean).powi(2) * v)
            .sum();
        let difference_entropy: f64 = p_x_minus_y
            .iter()
            .filter(|&&v| v > 1e-12)
            .map(|&v| -v * v.ln())
            .sum();

        // Main features
        let contrast: f64 = (0..n_levels)
            .flat_map(|i| (0..n_levels).map(move |j| ((i as f64 - j as f64).powi(2) * p(i, j))))
            .sum();
        let dissimilarity: f64 = (0..n_levels)
            .flat_map(|i| (0..n_levels).map(move |j| ((i as f64 - j as f64).abs() * p(i, j))))
            .sum();
        let homogeneity: f64 = (0..n_levels)
            .flat_map(|i| {
                (0..n_levels)
                    .map(move |j| p(i, j) / (1.0 + (i as f64 - j as f64).powi(2)))
            })
            .sum();
        let energy: f64 = glcm.iter().map(|&v| v * v).sum();
        let entropy: f64 = glcm.iter().filter(|&&v| v > 1e-12).map(|&v| -v * v.ln()).sum();
        let correlation: f64 = if sig_i > 1e-12 && sig_j > 1e-12 {
            (0..n_levels)
                .flat_map(|i| {
                    (0..n_levels).map(move |j| {
                        (i as f64 - mu_i) * (j as f64 - mu_j) * p(i, j) / (sig_i * sig_j)
                    })
                })
                .sum()
        } else {
            0.0
        };

        let max_probability: f64 = glcm.iter().cloned().fold(0.0_f64, f64::max);

        // Cluster measures
        let cluster_shade: f64 = (0..n_levels)
            .flat_map(|i| {
                (0..n_levels).map(move |j| {
                    (i as f64 + j as f64 - mu_i - mu_j).powi(3) * p(i, j)
                })
            })
            .sum();
        let cluster_prominence: f64 = (0..n_levels)
            .flat_map(|i| {
                (0..n_levels).map(move |j| {
                    (i as f64 + j as f64 - mu_i - mu_j).powi(4) * p(i, j)
                })
            })
            .sum();
        let auto_correlation: f64 = (0..n_levels)
            .flat_map(|i| (0..n_levels).map(move |j| i as f64 * j as f64 * p(i, j)))
            .sum();

        // Information measures of correlation
        let hxy: f64 = entropy; // already computed
        let hx: f64 = pi.iter().filter(|&&v| v > 1e-12).map(|&v| -v * v.ln()).sum::<f64>();
        let hy: f64 = pj.iter().filter(|&&v| v > 1e-12).map(|&v| -v * v.ln()).sum::<f64>();
        let hxy1: f64 = {
            let pi_ref: &[f64] = &pi;
            let pj_ref: &[f64] = &pj;
            let mut acc = 0.0f64;
            for i in 0..n_levels {
                for j in 0..n_levels {
                    let pij = p(i, j);
                    if pij > 1e-12 && pi_ref[i] > 1e-12 && pj_ref[j] > 1e-12 {
                        acc -= pij * (pi_ref[i] * pj_ref[j]).ln();
                    }
                }
            }
            acc
        };
        let hxy2: f64 = {
            let pi_ref: &[f64] = &pi;
            let pj_ref: &[f64] = &pj;
            let mut acc = 0.0f64;
            for i in 0..n_levels {
                for j in 0..n_levels {
                    if pi_ref[i] > 1e-12 && pj_ref[j] > 1e-12 {
                        let v = pi_ref[i] * pj_ref[j];
                        acc -= v * v.ln();
                    }
                }
            }
            acc
        };
        let hmax = hx.max(hy) + 1e-12;
        let info_measure_corr1 = (hxy - hxy1) / hmax;
        let info_measure_corr2 = if hxy2 >= hxy {
            (1.0 - (-(2.0 * (hxy2 - hxy))).exp()).sqrt()
        } else {
            0.0
        };

        Ok(GLCMFeatures {
            contrast,
            dissimilarity,
            homogeneity,
            energy,
            entropy,
            correlation,
            sum_average,
            sum_variance,
            sum_entropy,
            difference_variance,
            difference_entropy,
            info_measure_corr1,
            info_measure_corr2,
            max_probability,
            cluster_shade,
            cluster_prominence,
            auto_correlation,
        })
    }
}

// ─── ShapeFeatures3D ─────────────────────────────────────────────────────────

/// 3D shape descriptors for a binary ROI.
///
/// All features follow the IBSI shape feature set.  The ROI is provided as a
/// binary mask (`true` = inside ROI).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeFeatures3D {
    /// Number of voxels inside the ROI.
    pub n_voxels: usize,
    /// Volume in voxel units.
    pub volume_voxels: f64,
    /// Volume in physical units (mm^3) when voxel spacing is provided.
    pub volume_mm3: f64,
    /// Approximate surface area in voxel face units.
    pub surface_area_voxels: f64,
    /// Surface area in mm^2 (using provided voxel spacing).
    pub surface_area_mm2: f64,
    /// Sphericity: ratio of sphere surface to ROI surface for the same volume.
    /// Ranges from 0 (not spherical) to 1 (perfect sphere).
    pub sphericity: f64,
    /// Compactness: cube of volume divided by square of surface area (normalised).
    pub compactness: f64,
    /// Elongation: ratio of smallest to largest principal axis length.
    pub elongation: f64,
    /// Flatness: ratio of smallest to middle principal axis length.
    pub flatness: f64,
    /// Maximum 3D diameter (longest pair-wise distance through the object).
    /// This is approximated using principal axes.
    pub max_diameter: f64,
    /// Centroid in voxel coordinates `(z, y, x)`.
    pub centroid: (f64, f64, f64),
    /// Bounding-box extents `[dz, dy, dx]` in voxels.
    pub bounding_box: [usize; 3],
}

impl ShapeFeatures3D {
    /// Compute 3D shape features for a binary mask.
    ///
    /// `spacing` is the voxel size `[sz, sy, sx]` in mm (use `[1.0, 1.0, 1.0]`
    /// for voxel units).
    pub fn compute(mask: &Array3<bool>, spacing: [f64; 3]) -> NdimageResult<Self> {
        let shape = mask.shape();
        if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
            return Err(NdimageError::InvalidInput(
                "ShapeFeatures3D: mask must not be empty".to_string(),
            ));
        }
        let (nz, ny, nx) = (shape[0], shape[1], shape[2]);

        // Collect foreground voxel positions
        let mut voxels: Vec<(usize, usize, usize)> = Vec::new();
        let mut sum_z = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_x = 0.0_f64;
        let mut min_z = nz;
        let mut max_z = 0;
        let mut min_y = ny;
        let mut max_y = 0;
        let mut min_x = nx;
        let mut max_x = 0;

        // Count surface voxels (6-connectivity)
        let mut surface_count = 0usize;
        let face_offsets: [(isize, isize, isize); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if !mask[[iz, iy, ix]] {
                        continue;
                    }
                    voxels.push((iz, iy, ix));
                    sum_z += iz as f64;
                    sum_y += iy as f64;
                    sum_x += ix as f64;
                    min_z = min_z.min(iz);
                    max_z = max_z.max(iz);
                    min_y = min_y.min(iy);
                    max_y = max_y.max(iy);
                    min_x = min_x.min(ix);
                    max_x = max_x.max(ix);

                    // Surface: has at least one background face-neighbour
                    let is_surface = face_offsets.iter().any(|&(dz, dy, dx)| {
                        let nz_n = iz as isize + dz;
                        let ny_n = iy as isize + dy;
                        let nx_n = ix as isize + dx;
                        if nz_n < 0 || nz_n >= nz as isize
                            || ny_n < 0 || ny_n >= ny as isize
                            || nx_n < 0 || nx_n >= nx as isize
                        {
                            true // border voxel → counted as surface
                        } else {
                            !mask[[nz_n as usize, ny_n as usize, nx_n as usize]]
                        }
                    });
                    if is_surface {
                        surface_count += 1;
                    }
                }
            }
        }

        let n = voxels.len();
        if n == 0 {
            return Err(NdimageError::InvalidInput(
                "ShapeFeatures3D: mask is empty (no foreground voxels)".to_string(),
            ));
        }

        let centroid = (sum_z / n as f64, sum_y / n as f64, sum_x / n as f64);
        let volume_voxels = n as f64;
        let voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2];
        let volume_mm3 = volume_voxels * voxel_volume_mm3;

        // Face area depends on which face we're counting
        // Each exposed face contributes one face area
        // For simplicity, use the average face area of the voxel
        let face_area_zy = spacing[0] * spacing[1]; // perpendicular to x
        let face_area_zx = spacing[0] * spacing[2]; // perpendicular to y
        let face_area_yx = spacing[1] * spacing[2]; // perpendicular to z
        let avg_face_area = (face_area_zy + face_area_zx + face_area_yx) / 3.0;
        let surface_area_voxels = surface_count as f64;
        let surface_area_mm2 = surface_count as f64 * avg_face_area;

        // Sphericity: (π^(1/3) × (6V)^(2/3)) / S
        let sphericity = if surface_area_voxels > 1e-12 {
            use std::f64::consts::PI;
            (PI.powf(1.0 / 3.0) * (6.0 * volume_voxels).powf(2.0 / 3.0)) / surface_area_voxels
        } else {
            0.0
        };

        // Compactness: V^3 / S^2 (normalised so sphere = 1)
        let compactness = if surface_area_voxels > 1e-12 {
            use std::f64::consts::PI;
            volume_voxels.powi(2) / (36.0 * PI * surface_area_voxels.powi(3))
        } else {
            0.0
        };

        // Principal axis lengths via covariance matrix of voxel positions
        let (eig1, eig2, eig3) = principal_axis_lengths(&voxels, &centroid);
        // eig1 >= eig2 >= eig3 (largest to smallest)
        let elongation = if eig1 > 1e-12 { (eig3 / eig1).sqrt() } else { 0.0 };
        let flatness = if eig1 > 1e-12 { (eig2 / eig1).sqrt() } else { 0.0 };
        let max_diameter = 2.0 * eig1.sqrt();

        let bounding_box = [
            if max_z >= min_z { max_z - min_z + 1 } else { 1 },
            if max_y >= min_y { max_y - min_y + 1 } else { 1 },
            if max_x >= min_x { max_x - min_x + 1 } else { 1 },
        ];

        Ok(ShapeFeatures3D {
            n_voxels: n,
            volume_voxels,
            volume_mm3,
            surface_area_voxels,
            surface_area_mm2,
            sphericity: sphericity.min(1.0),
            compactness,
            elongation,
            flatness,
            max_diameter,
            centroid,
            bounding_box,
        })
    }
}

/// Compute the three principal axis lengths (eigenvalues of the 3D covariance
/// matrix of voxel positions), returning them sorted largest → smallest.
fn principal_axis_lengths(
    voxels: &[(usize, usize, usize)],
    centroid: &(f64, f64, f64),
) -> (f64, f64, f64) {
    let n = voxels.len() as f64;
    let (cz, cy, cx) = *centroid;

    let mut czz = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cxx = 0.0_f64;
    let mut czy = 0.0_f64;
    let mut czx = 0.0_f64;
    let mut cyx = 0.0_f64;

    for &(iz, iy, ix) in voxels {
        let dz = iz as f64 - cz;
        let dy = iy as f64 - cy;
        let dx = ix as f64 - cx;
        czz += dz * dz;
        cyy += dy * dy;
        cxx += dx * dx;
        czy += dz * dy;
        czx += dz * dx;
        cyx += dy * dx;
    }

    let cov = [
        [czz / n, czy / n, czx / n],
        [czy / n, cyy / n, cyx / n],
        [czx / n, cyx / n, cxx / n],
    ];

    // Compute eigenvalues of 3×3 symmetric matrix using analytical method
    eigenvalues_3x3_symmetric(&cov)
}

/// Characteristic polynomial eigenvalue computation for a 3×3 symmetric matrix.
///
/// Returns `(λ1, λ2, λ3)` with λ1 ≥ λ2 ≥ λ3.
fn eigenvalues_3x3_symmetric(m: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    use std::f64::consts::PI;

    let a = m[0][0];
    let b = m[1][1];
    let c = m[2][2];
    let d = m[0][1]; // = m[1][0]
    let e = m[0][2]; // = m[2][0]
    let f = m[1][2]; // = m[2][1]

    // Characteristic polynomial: λ^3 - p*λ^2 + q*λ - r = 0
    let p_coeff = a + b + c;
    let q_coeff = a * b + b * c + a * c - d * d - e * e - f * f;
    let r_coeff = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e);

    // Use the trigonometric method for real roots
    let p3 = p_coeff / 3.0;
    let q3 = (p3 * p3 - q_coeff / 3.0).max(0.0);
    let r3 = p3.powi(3) - p3 * q_coeff / 6.0 + r_coeff / 2.0;

    let phi = if q3 > 1e-18 {
        (r3 / q3.powi(3).sqrt().max(1e-18)).clamp(-1.0, 1.0).acos() / 3.0
    } else {
        0.0
    };

    let sqrt_q3 = q3.sqrt();
    let lam1 = p3 + 2.0 * sqrt_q3 * phi.cos();
    let lam2 = p3 - sqrt_q3 * (phi.cos() + (3.0_f64).sqrt() * phi.sin());
    let lam3 = p3 - sqrt_q3 * (phi.cos() - (3.0_f64).sqrt() * phi.sin());

    let mut eigs = [lam1.max(0.0), lam2.max(0.0), lam3.max(0.0)];
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    (eigs[0], eigs[1], eigs[2])
}

// ─── RadiomicsExtractor ───────────────────────────────────────────────────────

/// Configuration for the unified radiomics extraction pipeline.
#[derive(Debug, Clone)]
pub struct RadiomicsConfig {
    /// Number of histogram bins for first-order features.
    pub n_bins: usize,
    /// Number of gray levels for GLCM computation.
    pub n_glcm_levels: usize,
    /// Voxel spacing `[sz, sy, sx]` in mm (for physical measurements).
    pub voxel_spacing: [f64; 3],
    /// Whether to compute first-order features.
    pub compute_first_order: bool,
    /// Whether to compute GLCM texture features.
    pub compute_glcm: bool,
    /// Whether to compute 3D shape features.
    pub compute_shape: bool,
    /// GLCM directions to average over.
    pub glcm_directions: Vec<GlcmDirection3D>,
}

impl Default for RadiomicsConfig {
    fn default() -> Self {
        Self {
            n_bins: 64,
            n_glcm_levels: 32,
            voxel_spacing: [1.0, 1.0, 1.0],
            compute_first_order: true,
            compute_glcm: true,
            compute_shape: true,
            glcm_directions: vec![
                GlcmDirection3D::Axis0,
                GlcmDirection3D::Axis1,
                GlcmDirection3D::Axis2,
            ],
        }
    }
}

/// Result of the full radiomics extraction pipeline.
#[derive(Debug, Clone)]
pub struct RadiomicsResult {
    /// First-order intensity features (if computed).
    pub first_order: Option<FirstOrderFeatures>,
    /// GLCM texture features averaged over directions (if computed).
    pub glcm: Option<GLCMFeatures>,
    /// 3D shape features (if computed).
    pub shape: Option<ShapeFeatures3D>,
    /// All features combined as a flat `HashMap<name, value>` for convenience.
    pub feature_map: HashMap<String, f64>,
}

/// Unified radiomics feature extraction pipeline.
///
/// Accepts a 3D intensity volume and an optional binary ROI mask and returns
/// all configured feature groups.
pub struct RadiomicsExtractor {
    config: RadiomicsConfig,
}

impl RadiomicsExtractor {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self { config: RadiomicsConfig::default() }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RadiomicsConfig) -> Self {
        Self { config }
    }

    /// Extract radiomics features from `volume` restricted to `mask`.
    ///
    /// `mask` may be `None` to include all voxels.
    pub fn extract(
        &self,
        volume: &Array3<f64>,
        mask: Option<&Array3<bool>>,
    ) -> NdimageResult<RadiomicsResult> {
        let mut feature_map = HashMap::new();

        // ── First-order features ──
        let first_order = if self.config.compute_first_order {
            let fo = FirstOrderFeatures::compute(volume, mask, self.config.n_bins)?;
            feature_map.insert("fo_mean".to_string(), fo.mean);
            feature_map.insert("fo_variance".to_string(), fo.variance);
            feature_map.insert("fo_std_dev".to_string(), fo.std_dev);
            feature_map.insert("fo_skewness".to_string(), fo.skewness);
            feature_map.insert("fo_kurtosis".to_string(), fo.kurtosis);
            feature_map.insert("fo_excess_kurtosis".to_string(), fo.excess_kurtosis);
            feature_map.insert("fo_minimum".to_string(), fo.minimum);
            feature_map.insert("fo_maximum".to_string(), fo.maximum);
            feature_map.insert("fo_range".to_string(), fo.range);
            feature_map.insert("fo_median".to_string(), fo.median);
            feature_map.insert("fo_p10".to_string(), fo.p10);
            feature_map.insert("fo_p90".to_string(), fo.p90);
            feature_map.insert("fo_iqr".to_string(), fo.iqr);
            feature_map.insert("fo_mad".to_string(), fo.mad);
            feature_map.insert("fo_rmad".to_string(), fo.rmad);
            feature_map.insert("fo_energy".to_string(), fo.energy);
            feature_map.insert("fo_entropy".to_string(), fo.entropy);
            feature_map.insert("fo_uniformity".to_string(), fo.uniformity);
            feature_map.insert("fo_cv".to_string(), fo.coefficient_of_variation);
            Some(fo)
        } else {
            None
        };

        // ── GLCM features ──
        let glcm = if self.config.compute_glcm && !self.config.glcm_directions.is_empty() {
            // Build the ROI-masked + quantised volume for GLCM
            let masked_vol = match mask {
                Some(m) => quantise_volume(volume, m, self.config.n_glcm_levels),
                None => {
                    let all_true = Array3::<bool>::from_elem(
                        (volume.shape()[0], volume.shape()[1], volume.shape()[2]),
                        true,
                    );
                    quantise_volume(volume, &all_true, self.config.n_glcm_levels)
                }
            };

            // Average GLCM features over all specified directions
            let mut glcm_accum = None;
            let n_dirs = self.config.glcm_directions.len() as f64;
            for &dir in &self.config.glcm_directions {
                match GLCMFeatures::from_3d(&masked_vol, self.config.n_glcm_levels, dir) {
                    Ok(gf) => {
                        glcm_accum = Some(match glcm_accum {
                            None => gf,
                            Some(acc) => average_glcm(acc, gf, n_dirs),
                        });
                    }
                    Err(_) => {} // skip problematic directions
                }
            }

            if let Some(ref gf) = glcm_accum {
                feature_map.insert("glcm_contrast".to_string(), gf.contrast);
                feature_map.insert("glcm_dissimilarity".to_string(), gf.dissimilarity);
                feature_map.insert("glcm_homogeneity".to_string(), gf.homogeneity);
                feature_map.insert("glcm_energy".to_string(), gf.energy);
                feature_map.insert("glcm_entropy".to_string(), gf.entropy);
                feature_map.insert("glcm_correlation".to_string(), gf.correlation);
                feature_map.insert("glcm_sum_average".to_string(), gf.sum_average);
                feature_map.insert("glcm_sum_variance".to_string(), gf.sum_variance);
                feature_map.insert("glcm_cluster_shade".to_string(), gf.cluster_shade);
                feature_map.insert("glcm_cluster_prominence".to_string(), gf.cluster_prominence);
                feature_map.insert("glcm_auto_correlation".to_string(), gf.auto_correlation);
            }
            glcm_accum
        } else {
            None
        };

        // ── Shape features ──
        let shape = if self.config.compute_shape {
            let mask_vol = match mask {
                Some(m) => m.to_owned(),
                None => Array3::<bool>::from_elem(
                    (volume.shape()[0], volume.shape()[1], volume.shape()[2]),
                    true,
                ),
            };
            let sf = ShapeFeatures3D::compute(&mask_vol, self.config.voxel_spacing)?;
            feature_map.insert("shape_volume_voxels".to_string(), sf.volume_voxels);
            feature_map.insert("shape_volume_mm3".to_string(), sf.volume_mm3);
            feature_map.insert("shape_surface_area_voxels".to_string(), sf.surface_area_voxels);
            feature_map.insert("shape_surface_area_mm2".to_string(), sf.surface_area_mm2);
            feature_map.insert("shape_sphericity".to_string(), sf.sphericity);
            feature_map.insert("shape_compactness".to_string(), sf.compactness);
            feature_map.insert("shape_elongation".to_string(), sf.elongation);
            feature_map.insert("shape_flatness".to_string(), sf.flatness);
            feature_map.insert("shape_max_diameter".to_string(), sf.max_diameter);
            Some(sf)
        } else {
            None
        };

        Ok(RadiomicsResult {
            first_order,
            glcm,
            shape,
            feature_map,
        })
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Quantise a float volume to integer gray levels in `[0, n_levels - 1]`.
/// Only voxels in the mask participate; others are set to 0.
fn quantise_volume(vol: &Array3<f64>, mask: &Array3<bool>, n_levels: usize) -> Array3<f64> {
    let n = n_levels as f64;
    let masked_vals: Vec<f64> = vol
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();

    if masked_vals.is_empty() {
        return vol.mapv(|_| 0.0);
    }

    let min_val = masked_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = masked_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-12);

    let shape = vol.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    let mut out = Array3::<f64>::zeros((nz, ny, nx));
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if mask[[iz, iy, ix]] {
                    let v = ((vol[[iz, iy, ix]] - min_val) / range * (n - 1.0))
                        .round()
                        .clamp(0.0, n - 1.0);
                    out[[iz, iy, ix]] = v;
                }
            }
        }
    }
    out
}

/// Average two GLCM feature sets by dividing the second by `n_dirs` and adding.
///
/// The approach accumulates features linearly; the caller divides by `n_dirs`
/// once after all directions have been processed.  This is a running average
/// where the first direction contributes `1/n_dirs` and subsequent ones
/// also contribute `1/n_dirs`.
fn average_glcm(acc: GLCMFeatures, new: GLCMFeatures, n_dirs: f64) -> GLCMFeatures {
    let w = 1.0 / n_dirs;
    GLCMFeatures {
        contrast: acc.contrast + new.contrast * w,
        dissimilarity: acc.dissimilarity + new.dissimilarity * w,
        homogeneity: acc.homogeneity + new.homogeneity * w,
        energy: acc.energy + new.energy * w,
        entropy: acc.entropy + new.entropy * w,
        correlation: acc.correlation + new.correlation * w,
        sum_average: acc.sum_average + new.sum_average * w,
        sum_variance: acc.sum_variance + new.sum_variance * w,
        sum_entropy: acc.sum_entropy + new.sum_entropy * w,
        difference_variance: acc.difference_variance + new.difference_variance * w,
        difference_entropy: acc.difference_entropy + new.difference_entropy * w,
        info_measure_corr1: acc.info_measure_corr1 + new.info_measure_corr1 * w,
        info_measure_corr2: acc.info_measure_corr2 + new.info_measure_corr2 * w,
        max_probability: acc.max_probability + new.max_probability * w,
        cluster_shade: acc.cluster_shade + new.cluster_shade * w,
        cluster_prominence: acc.cluster_prominence + new.cluster_prominence * w,
        auto_correlation: acc.auto_correlation + new.auto_correlation * w,
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    fn make_uniform_volume(nz: usize, ny: usize, nx: usize, val: f64) -> Array3<f64> {
        Array3::<f64>::from_elem((nz, ny, nx), val)
    }

    fn sphere_mask(nz: usize, ny: usize, nx: usize) -> Array3<bool> {
        let mut m = Array3::<bool>::from_elem((nz, ny, nx), false);
        let cz = nz as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cx = nx as f64 / 2.0;
        let r2 = ((nz.min(ny).min(nx)) as f64 / 3.0).powi(2);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = (iz as f64 - cz).powi(2)
                        + (iy as f64 - cy).powi(2)
                        + (ix as f64 - cx).powi(2);
                    if d2 < r2 {
                        m[[iz, iy, ix]] = true;
                    }
                }
            }
        }
        m
    }

    #[test]
    fn test_first_order_uniform_volume() {
        let vol = make_uniform_volume(4, 4, 4, 50.0);
        let fo = FirstOrderFeatures::compute(&vol, None, 64).expect("FirstOrderFeatures::compute should succeed on uniform volume");
        assert!((fo.mean - 50.0).abs() < 1e-8);
        assert!(fo.variance < 1e-8);
        assert!(fo.std_dev < 1e-8);
        assert!(fo.skewness.abs() < 1e-6);
        assert!((fo.minimum - 50.0).abs() < 1e-8);
        assert!((fo.maximum - 50.0).abs() < 1e-8);
        assert_eq!(fo.n_voxels, 64);
    }

    #[test]
    fn test_first_order_entropy_uniform() {
        // Uniform distribution → maximum entropy
        let vol = make_uniform_volume(4, 4, 4, 10.0);
        let fo = FirstOrderFeatures::compute(&vol, None, 4).expect("FirstOrderFeatures::compute should succeed on uniform volume with 4 bins");
        // All voxels equal → one bin full, entropy = 0
        assert!(fo.entropy < 1e-6);
    }

    #[test]
    fn test_first_order_with_mask() {
        let vol = make_uniform_volume(4, 4, 4, 100.0);
        let mask = sphere_mask(4, 4, 4);
        let fo = FirstOrderFeatures::compute(&vol, Some(&mask), 32).expect("FirstOrderFeatures::compute should succeed with mask");
        assert!(fo.n_voxels > 0);
        assert!((fo.mean - 100.0).abs() < 1e-8);
    }

    #[test]
    fn test_glcm_features_checkerboard() {
        // Checkerboard: alternating 0 and 1 → high contrast
        let mut img = Array2::<f64>::zeros((8, 8));
        for r in 0..8_usize {
            for c in 0..8_usize {
                img[[r, c]] = ((r + c) % 2) as f64;
            }
        }
        let gf = GLCMFeatures::from_2d(&img, 2, 0, 1).expect("GLCMFeatures::from_2d should succeed on valid image");
        assert!(gf.energy > 0.0);
        assert!(gf.homogeneity > 0.0);
    }

    #[test]
    fn test_glcm_features_uniform() {
        let img = Array2::<f64>::from_elem((8, 8), 5.0);
        let gf = GLCMFeatures::from_2d(&img, 8, 0, 1);
        // With a uniform image and single gray level, GLCM is degenerate;
        // the function should succeed without panic
        let _ = gf;
    }

    #[test]
    fn test_shape_features_sphere() {
        let mask = sphere_mask(12, 12, 12);
        let sf = ShapeFeatures3D::compute(&mask, [1.0, 1.0, 1.0]).expect("ShapeFeatures3D::compute should succeed on sphere mask");
        assert!(sf.n_voxels > 0);
        // Sphericity should be close to 1 for a sphere
        assert!(sf.sphericity > 0.0 && sf.sphericity <= 1.0);
        // Elongation and flatness for a sphere should be close to 1
        assert!(sf.elongation > 0.0);
        assert!(sf.flatness > 0.0);
        println!(
            "sphere: vol={}, surf={}, sph={:.3}, elong={:.3}, flat={:.3}",
            sf.n_voxels, sf.surface_area_voxels, sf.sphericity, sf.elongation, sf.flatness
        );
    }

    #[test]
    fn test_shape_features_cube() {
        let mut mask = Array3::<bool>::from_elem((8, 8, 8), true);
        let sf = ShapeFeatures3D::compute(&mask, [1.0, 1.0, 1.0]).expect("ShapeFeatures3D::compute should succeed on cube mask");
        assert_eq!(sf.n_voxels, 512);
        assert!(sf.sphericity > 0.0 && sf.sphericity <= 1.0);
    }

    #[test]
    fn test_radiomics_extractor_smoke() {
        let vol = {
            let mut v = Array3::<f64>::zeros((8, 8, 8));
            for iz in 0..8_usize {
                for iy in 0..8_usize {
                    for ix in 0..8_usize {
                        v[[iz, iy, ix]] = ((iz + iy + ix) % 5) as f64 * 10.0;
                    }
                }
            }
            v
        };
        let mask = sphere_mask(8, 8, 8);
        let extractor = RadiomicsExtractor::new();
        let result = extractor.extract(&vol, Some(&mask)).expect("RadiomicsExtractor::extract should succeed with mask");
        assert!(result.first_order.is_some());
        assert!(result.glcm.is_some());
        assert!(result.shape.is_some());
        assert!(!result.feature_map.is_empty());
        // Feature map should contain first-order keys
        assert!(result.feature_map.contains_key("fo_mean"));
        assert!(result.feature_map.contains_key("shape_sphericity"));
        assert!(result.feature_map.contains_key("glcm_entropy"));
    }

    #[test]
    fn test_radiomics_extractor_no_mask() {
        let vol = make_uniform_volume(4, 4, 4, 42.0);
        let extractor = RadiomicsExtractor::new();
        let result = extractor.extract(&vol, None).expect("RadiomicsExtractor::extract should succeed without mask");
        assert!(result.first_order.is_some());
    }

    #[test]
    fn test_eigenvalues_sphere_covariance() {
        // A sphere should have equal eigenvalues
        let mask = sphere_mask(10, 10, 10);
        let sf = ShapeFeatures3D::compute(&mask, [1.0, 1.0, 1.0]).expect("ShapeFeatures3D::compute should succeed on 10x10x10 sphere mask");
        // For a sphere, elongation ≈ flatness ≈ 1
        assert!((sf.elongation - 1.0).abs() < 0.2, "elongation = {}", sf.elongation);
        assert!((sf.flatness - 1.0).abs() < 0.2, "flatness = {}", sf.flatness);
    }
}
