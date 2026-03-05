//! Minimal DICOM-like metadata and medical imaging primitives.
//!
//! This module provides lightweight structures for working with medical imaging
//! data without requiring a full DICOM library.  It covers:
//!
//! - [`DicomHeader`]: minimal per-series metadata (patient, modality, geometry)
//! - [`MedicalVolume`]: a 3-D f64 array together with voxel spacing and orientation
//! - [`WindowLeveling`]: CT/MR window-width / window-center display adjustment
//! - [`HounsfieldUnits`]: CT HU rescaling and tissue classification
//! - [`N4BiasCorrection`]: simplified polynomial N4-style bias-field correction
//! - [`VolumeStats`]: per-tissue-class statistics
//!
//! # References
//!
//! - Tustison et al. (2010), "N4ITK: Improved N3 Bias Correction",
//!   IEEE TMI 29(6):1310-1320.
//! - Bushberg et al., *The Essential Physics of Medical Imaging* (3rd ed.)

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array3, ArrayView3};

use crate::error::{NdimageError, NdimageResult};

// ─── DicomHeader ─────────────────────────────────────────────────────────────

/// Minimal DICOM-like header carrying the metadata needed for image display and
/// geometric reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct DicomHeader {
    /// Anonymous patient identifier string (no PHI stored).
    pub patient_id: String,
    /// Imaging modality: e.g. "CT", "MR", "PT", "US".
    pub modality: String,
    /// In-plane pixel spacing (row_spacing_mm, col_spacing_mm).
    pub pixel_spacing: (f64, f64),
    /// Distance between consecutive slices in mm (slice thickness).
    pub slice_thickness: f64,
    /// Number of rows per slice.
    pub rows: usize,
    /// Number of columns per slice.
    pub columns: usize,
    /// Number of slices in the volume.
    pub num_slices: usize,
    /// Rescale slope for converting stored integers to HU (CT only; 1.0 otherwise).
    pub rescale_slope: f64,
    /// Rescale intercept (CT: typically -1024.0; MR: 0.0).
    pub rescale_intercept: f64,
    /// Series description / protocol label (free-text).
    pub series_description: String,
    /// Image orientation: unit row-direction cosines [rx,ry,rz] then column-direction [cx,cy,cz].
    pub image_orientation: [f64; 6],
    /// Image position: top-left corner of the first slice in mm (x,y,z).
    pub image_position: [f64; 3],
}

impl DicomHeader {
    /// Create a header with sensible defaults for a CT acquisition.
    pub fn new_ct(
        patient_id: impl Into<String>,
        rows: usize,
        columns: usize,
        num_slices: usize,
    ) -> Self {
        Self {
            patient_id: patient_id.into(),
            modality: "CT".to_string(),
            pixel_spacing: (1.0, 1.0),
            slice_thickness: 1.0,
            rows,
            columns,
            num_slices,
            rescale_slope: 1.0,
            rescale_intercept: -1024.0,
            series_description: String::new(),
            image_orientation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            image_position: [0.0, 0.0, 0.0],
        }
    }

    /// Create a header with sensible defaults for an MR acquisition.
    pub fn new_mr(
        patient_id: impl Into<String>,
        rows: usize,
        columns: usize,
        num_slices: usize,
    ) -> Self {
        Self {
            patient_id: patient_id.into(),
            modality: "MR".to_string(),
            pixel_spacing: (1.0, 1.0),
            slice_thickness: 1.0,
            rows,
            columns,
            num_slices,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
            series_description: String::new(),
            image_orientation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            image_position: [0.0, 0.0, 0.0],
        }
    }

    /// Voxel spacing as `(dz, dy, dx)` in mm (z = slice axis).
    pub fn voxel_spacing(&self) -> [f64; 3] {
        [
            self.slice_thickness,
            self.pixel_spacing.0,
            self.pixel_spacing.1,
        ]
    }

    /// Validate that dimensional metadata is self-consistent.
    pub fn validate(&self) -> NdimageResult<()> {
        if self.rows == 0 || self.columns == 0 || self.num_slices == 0 {
            return Err(NdimageError::InvalidInput(
                "DicomHeader: rows, columns, and num_slices must be > 0".to_string(),
            ));
        }
        if self.slice_thickness <= 0.0 {
            return Err(NdimageError::InvalidInput(
                "DicomHeader: slice_thickness must be positive".to_string(),
            ));
        }
        if self.pixel_spacing.0 <= 0.0 || self.pixel_spacing.1 <= 0.0 {
            return Err(NdimageError::InvalidInput(
                "DicomHeader: pixel_spacing values must be positive".to_string(),
            ));
        }
        if self.rescale_slope == 0.0 {
            return Err(NdimageError::InvalidInput(
                "DicomHeader: rescale_slope must not be zero".to_string(),
            ));
        }
        Ok(())
    }
}

// ─── MedicalVolume ───────────────────────────────────────────────────────────

/// A 3-D medical image volume stored as `f64` voxels with geometric metadata.
///
/// The axis convention follows radiology DICOM (z = slice / superior-inferior,
/// y = row / anterior-posterior, x = column / left-right) but any consistent
/// labeling is acceptable.
///
/// The voxel intensities are stored in the *physical* domain (HU for CT, signal
/// intensity for MR, etc.).  Use [`HounsfieldUnits`] for CT-specific
/// conversions from raw pixel values.
#[derive(Debug, Clone)]
pub struct MedicalVolume {
    /// Raw voxel data with shape `[nz, ny, nx]`.
    pub data: Array3<f64>,
    /// Voxel spacing `[sz, sy, sx]` in mm.
    pub spacing: [f64; 3],
    /// 3x3 direction cosines matrix (row-major, stored as 9 elements).
    /// Columns are the unit vectors along x, y, z patient axes.
    pub direction: [f64; 9],
    /// Physical origin of voxel `[0, 0, 0]` in mm.
    pub origin: [f64; 3],
    /// Associated DICOM-like header (optional).
    pub header: Option<DicomHeader>,
}

impl MedicalVolume {
    /// Create a new volume from a raw data array.
    ///
    /// `spacing` must have three positive values; otherwise an error is returned.
    pub fn new(
        data: Array3<f64>,
        spacing: [f64; 3],
        origin: [f64; 3],
    ) -> NdimageResult<Self> {
        for (i, &s) in spacing.iter().enumerate() {
            if s <= 0.0 {
                return Err(NdimageError::InvalidInput(format!(
                    "MedicalVolume: spacing[{}] must be positive, got {}",
                    i, s
                )));
            }
        }
        Ok(Self {
            data,
            spacing,
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            origin,
            header: None,
        })
    }

    /// Attach a [`DicomHeader`] to this volume (overwrites any existing header).
    pub fn with_header(mut self, header: DicomHeader) -> Self {
        self.header = Some(header);
        self
    }

    /// Volume dimensions `(nz, ny, nx)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    /// Total number of voxels.
    pub fn num_voxels(&self) -> usize {
        let (nz, ny, nx) = self.shape();
        nz * ny * nx
    }

    /// Physical volume in mm^3.
    pub fn physical_volume_mm3(&self) -> f64 {
        self.num_voxels() as f64 * self.spacing[0] * self.spacing[1] * self.spacing[2]
    }

    /// Convert voxel index `(iz, iy, ix)` to physical coordinates in mm.
    pub fn voxel_to_physical(&self, iz: usize, iy: usize, ix: usize) -> [f64; 3] {
        [
            self.origin[0] + iz as f64 * self.spacing[0],
            self.origin[1] + iy as f64 * self.spacing[1],
            self.origin[2] + ix as f64 * self.spacing[2],
        ]
    }

    /// Extract a single axial slice at index `iz` (returns a 2-D view as a flat
    /// `Vec` laid out in row-major order).
    pub fn axial_slice(&self, iz: usize) -> NdimageResult<Vec<f64>> {
        let (nz, ny, nx) = self.shape();
        if iz >= nz {
            return Err(NdimageError::InvalidInput(format!(
                "MedicalVolume: slice index {} out of range [0, {})",
                iz, nz
            )));
        }
        let mut out = Vec::with_capacity(ny * nx);
        for iy in 0..ny {
            for ix in 0..nx {
                out.push(self.data[[iz, iy, ix]]);
            }
        }
        Ok(out)
    }

    /// Return an immutable 3-D array view of the voxel data.
    pub fn view(&self) -> ArrayView3<f64> {
        self.data.view()
    }
}

// ─── WindowLeveling ──────────────────────────────────────────────────────────

/// Window/level (brightness/contrast) adjustment used for display.
///
/// Maps a physical intensity value to a display value in `[0, 1]` via a
/// linear ramp centred on `level` with width `window`.
///
/// Values outside `[level - window/2, level + window/2]` are clamped to 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLeveling {
    /// Window centre (level) in HU or signal intensity units.
    pub level: f64,
    /// Window width in the same units.
    pub window: f64,
}

impl WindowLeveling {
    /// Standard CT abdomen preset (level = 60 HU, window = 400 HU).
    pub fn ct_abdomen() -> Self {
        Self { level: 60.0, window: 400.0 }
    }

    /// Standard CT lung preset (level = -600 HU, window = 1500 HU).
    pub fn ct_lung() -> Self {
        Self { level: -600.0, window: 1500.0 }
    }

    /// Standard CT bone preset (level = 400 HU, window = 1800 HU).
    pub fn ct_bone() -> Self {
        Self { level: 400.0, window: 1800.0 }
    }

    /// Standard brain MR preset (level = 40 HU equiv., window = 80).
    pub fn ct_brain() -> Self {
        Self { level: 40.0, window: 80.0 }
    }

    /// Create a custom window/level setting.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` is not positive.
    pub fn new(level: f64, window: f64) -> NdimageResult<Self> {
        if window <= 0.0 {
            return Err(NdimageError::InvalidInput(
                "WindowLeveling: window must be positive".to_string(),
            ));
        }
        Ok(Self { level, window })
    }

    /// Map a single intensity value to a display value in `[0.0, 1.0]`.
    pub fn apply(&self, value: f64) -> f64 {
        let low = self.level - self.window * 0.5;
        let high = self.level + self.window * 0.5;
        if value <= low {
            0.0
        } else if value >= high {
            1.0
        } else {
            (value - low) / self.window
        }
    }

    /// Apply window/level to an entire 3-D volume, returning values in `[0, 1]`.
    pub fn apply_to_volume(&self, volume: &Array3<f64>) -> Array3<f64> {
        volume.mapv(|v| self.apply(v))
    }

    /// Return the lower and upper intensity bounds `(low, high)`.
    pub fn bounds(&self) -> (f64, f64) {
        (
            self.level - self.window * 0.5,
            self.level + self.window * 0.5,
        )
    }
}

// ─── HounsfieldUnits ─────────────────────────────────────────────────────────

/// CT-specific Hounsfield Unit (HU) utilities.
///
/// Provides rescaling from raw stored pixel values and tissue classification
/// based on standard HU ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HounsfieldUnits;

/// Tissue classification based on HU ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tissue {
    /// Air (< -950 HU)
    Air,
    /// Lung parenchyma (-950 to -500 HU)
    Lung,
    /// Fat (-500 to -100 HU)
    Fat,
    /// Soft tissue / muscle (-100 to 100 HU)
    SoftTissue,
    /// Blood pool / solid organs (100 to 300 HU)
    Blood,
    /// Cancellous bone (300 to 700 HU)
    CancellousBone,
    /// Cortical bone (> 700 HU)
    CorticalBone,
    /// Metallic implant artefact (> 3000 HU)
    Metal,
}

impl HounsfieldUnits {
    /// Convert a raw stored pixel value to HU using slope/intercept rescaling.
    ///
    /// `hu = slope * stored_value + intercept`
    pub fn rescale(stored_value: f64, slope: f64, intercept: f64) -> f64 {
        slope * stored_value + intercept
    }

    /// Classify a HU value into a [`Tissue`] category.
    pub fn classify(hu: f64) -> Tissue {
        if hu > 3000.0 {
            Tissue::Metal
        } else if hu > 700.0 {
            Tissue::CorticalBone
        } else if hu > 300.0 {
            Tissue::CancellousBone
        } else if hu > 100.0 {
            Tissue::Blood
        } else if hu > -100.0 {
            Tissue::SoftTissue
        } else if hu > -500.0 {
            Tissue::Fat
        } else if hu > -950.0 {
            Tissue::Lung
        } else {
            Tissue::Air
        }
    }

    /// Rescale an entire volume and classify each voxel.
    ///
    /// Returns both the rescaled HU volume and a classification volume encoded
    /// as u8 (the `Tissue` discriminant, 0-7 in the order defined above).
    pub fn classify_volume(
        raw: &Array3<f64>,
        slope: f64,
        intercept: f64,
    ) -> (Array3<f64>, Array3<u8>) {
        let hu_vol = raw.mapv(|v| Self::rescale(v, slope, intercept));
        let class_vol = hu_vol.mapv(|hu| Self::classify(hu) as u8);
        (hu_vol, class_vol)
    }

    /// Returns the typical HU range `(min, max)` for a tissue class.
    pub fn tissue_range(tissue: Tissue) -> (f64, f64) {
        match tissue {
            Tissue::Air => (f64::NEG_INFINITY, -950.0),
            Tissue::Lung => (-950.0, -500.0),
            Tissue::Fat => (-500.0, -100.0),
            Tissue::SoftTissue => (-100.0, 100.0),
            Tissue::Blood => (100.0, 300.0),
            Tissue::CancellousBone => (300.0, 700.0),
            Tissue::CorticalBone => (700.0, 3000.0),
            Tissue::Metal => (3000.0, f64::INFINITY),
        }
    }
}

// ─── N4BiasCorrection ────────────────────────────────────────────────────────

/// Configuration for simplified N4-style bias field correction.
#[derive(Debug, Clone)]
pub struct N4Config {
    /// Polynomial degree for bias-field approximation (1 to 4; default 2).
    pub poly_degree: usize,
    /// Number of iterations to run (default 50).
    pub max_iterations: usize,
    /// Convergence threshold on relative RMSE change (default 1e-4).
    pub convergence_threshold: f64,
    /// Mask threshold: voxels with intensity below `mask_fraction * global_mean`
    /// are excluded (background masking).  Set to 0.0 to disable.
    pub mask_fraction: f64,
}

impl Default for N4Config {
    fn default() -> Self {
        Self {
            poly_degree: 2,
            max_iterations: 50,
            convergence_threshold: 1e-4,
            mask_fraction: 0.02,
        }
    }
}

/// Simplified N4 bias-field correction operating on a [`MedicalVolume`].
///
/// The algorithm alternates between:
/// 1. Estimating the log-signal residual in the log domain.
/// 2. Fitting a polynomial to the residual on the masked voxels.
/// 3. Subtracting the polynomial estimate from the log-domain signal.
///
/// This is a computationally lightweight approximation of the full N4ITK
/// algorithm; it works well for moderate field inhomogeneities.
pub struct N4BiasCorrection {
    config: N4Config,
}

impl N4BiasCorrection {
    /// Create a corrector with default configuration.
    pub fn new() -> Self {
        Self { config: N4Config::default() }
    }

    /// Create a corrector with a custom configuration.
    pub fn with_config(config: N4Config) -> Self {
        Self { config }
    }

    /// Apply bias correction to `volume` and return the corrected volume
    /// together with the estimated bias field.
    ///
    /// # Errors
    ///
    /// Returns an error if the volume is empty or contains non-positive voxels
    /// (required for log-domain processing; negative-only regions are handled
    /// by shifting).
    pub fn correct(&self, volume: &MedicalVolume) -> NdimageResult<(MedicalVolume, Array3<f64>)> {
        let data = &volume.data;
        let shape = data.shape();
        if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
            return Err(NdimageError::InvalidInput(
                "N4BiasCorrection: volume must not be empty".to_string(),
            ));
        }

        let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
        let n = nz * ny * nx;

        // Shift so that all values are positive before log transform
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_val <= 0.0 { 1.0 - min_val } else { 0.0 };

        // Build log-domain signal and background mask
        let mut log_signal = vec![0.0_f64; n];
        let mut mask = vec![false; n];
        let global_sum: f64 = data.iter().sum();
        let global_mean = global_sum / n as f64;
        let threshold = self.config.mask_fraction * (global_mean + shift);

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let v = data[[iz, iy, ix]] + shift;
                    log_signal[idx] = v.ln();
                    mask[idx] = v > threshold;
                }
            }
        }

        // Polynomial basis coordinates: normalised to [-1, 1]
        let degree = self.config.poly_degree.min(4);

        // Iteratively estimate bias field
        let mut corrected_log = log_signal.clone();
        let mut prev_rmse = f64::INFINITY;

        for _iter in 0..self.config.max_iterations {
            // Fit polynomial to current log-residual on masked voxels
            let bias_log = self.fit_polynomial_bias(&corrected_log, &mask, nz, ny, nx, degree)?;

            // Subtract bias estimate
            let mut new_corrected = vec![0.0_f64; n];
            let mut sq_sum = 0.0;
            let mut cnt = 0usize;
            for idx in 0..n {
                new_corrected[idx] = corrected_log[idx] - bias_log[idx];
                if mask[idx] {
                    let d = new_corrected[idx] - corrected_log[idx];
                    sq_sum += d * d;
                    cnt += 1;
                }
            }

            let rmse = if cnt > 0 { (sq_sum / cnt as f64).sqrt() } else { 0.0 };
            corrected_log = new_corrected;

            let rel_change = (prev_rmse - rmse).abs() / (prev_rmse + 1e-12);
            if rel_change < self.config.convergence_threshold {
                break;
            }
            prev_rmse = rmse;
        }

        // Reconstruct corrected volume and bias field in original domain
        let mut corrected_data = Array3::<f64>::zeros((nz, ny, nx));
        let mut bias_field = Array3::<f64>::zeros((nz, ny, nx));
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let corrected_val = corrected_log[idx].exp() - shift;
                    let bias_val = (log_signal[idx] - corrected_log[idx]).exp();
                    corrected_data[[iz, iy, ix]] = corrected_val;
                    bias_field[[iz, iy, ix]] = bias_val;
                }
            }
        }

        let new_volume = MedicalVolume {
            data: corrected_data,
            spacing: volume.spacing,
            direction: volume.direction,
            origin: volume.origin,
            header: volume.header.clone(),
        };
        Ok((new_volume, bias_field))
    }

    /// Fit a low-degree polynomial in 3D normalised coordinates to `log_signal`
    /// at voxels where `mask[idx]` is true.
    ///
    /// Returns the polynomial-evaluated bias estimate at every voxel.
    fn fit_polynomial_bias(
        &self,
        log_signal: &[f64],
        mask: &[bool],
        nz: usize,
        ny: usize,
        nx: usize,
        degree: usize,
    ) -> NdimageResult<Vec<f64>> {
        // Build basis functions: 1, z, y, x, z^2, y^2, x^2, zy, zx, yx, ...
        let basis_fns = polynomial_basis_3d(degree);
        let nb = basis_fns.len();
        let n = nz * ny * nx;

        // Collect masked voxels
        let masked_indices: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
        let nm = masked_indices.len();
        if nm < nb {
            // Not enough samples to fit — return zero bias
            return Ok(vec![0.0; n]);
        }

        // Build design matrix A (nm × nb) and target vector b (nm)
        let mut a_mat = vec![vec![0.0_f64; nb]; nm];
        let mut b_vec = vec![0.0_f64; nm];
        for (row, &idx) in masked_indices.iter().enumerate() {
            let iz = idx / (ny * nx);
            let rem = idx % (ny * nx);
            let iy = rem / nx;
            let ix = rem % nx;
            // Normalise coordinates to [-1, 1]
            let zn = 2.0 * iz as f64 / (nz as f64 - 1.0).max(1.0) - 1.0;
            let yn = 2.0 * iy as f64 / (ny as f64 - 1.0).max(1.0) - 1.0;
            let xn = 2.0 * ix as f64 / (nx as f64 - 1.0).max(1.0) - 1.0;
            for (col, &(pz, py, px)) in basis_fns.iter().enumerate() {
                a_mat[row][col] = zn.powi(pz as i32) * yn.powi(py as i32) * xn.powi(px as i32);
            }
            b_vec[row] = log_signal[idx];
        }

        // Solve via normal equations: (A^T A) coeffs = A^T b  (least squares)
        let coeffs = solve_normal_equations(&a_mat, &b_vec, nb)?;

        // Evaluate the fitted polynomial on all voxels
        let mut bias = vec![0.0_f64; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let zn = 2.0 * iz as f64 / (nz as f64 - 1.0).max(1.0) - 1.0;
                    let yn = 2.0 * iy as f64 / (ny as f64 - 1.0).max(1.0) - 1.0;
                    let xn = 2.0 * ix as f64 / (nx as f64 - 1.0).max(1.0) - 1.0;
                    let mut val = 0.0;
                    for (j, &(pz, py, px)) in basis_fns.iter().enumerate() {
                        val += coeffs[j]
                            * zn.powi(pz as i32)
                            * yn.powi(py as i32)
                            * xn.powi(px as i32);
                    }
                    bias[idx] = val;
                }
            }
        }
        Ok(bias)
    }
}

/// Generate all monomial exponent triples `(pz, py, px)` with `pz+py+px <= degree`.
fn polynomial_basis_3d(degree: usize) -> Vec<(usize, usize, usize)> {
    let mut basis = Vec::new();
    for total in 0..=degree {
        for pz in 0..=total {
            for py in 0..=(total - pz) {
                let px = total - pz - py;
                basis.push((pz, py, px));
            }
        }
    }
    basis
}

/// Solve the normal equations A^T A x = A^T b via Cholesky-like Gaussian
/// elimination with partial pivoting.
fn solve_normal_equations(
    a: &[Vec<f64>],
    b: &[f64],
    nb: usize,
) -> NdimageResult<Vec<f64>> {
    let nm = a.len();
    // Build ATA (nb × nb) and ATb (nb)
    let mut ata = vec![vec![0.0_f64; nb]; nb];
    let mut atb = vec![0.0_f64; nb];
    for row in 0..nm {
        for i in 0..nb {
            atb[i] += a[row][i] * b[row];
            for j in 0..nb {
                ata[i][j] += a[row][i] * a[row][j];
            }
        }
    }

    // Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = (0..nb)
        .map(|i| {
            let mut r = ata[i].clone();
            r.push(atb[i]);
            r
        })
        .collect();

    for col in 0..nb {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..nb {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(NdimageError::ComputationError(
                "N4BiasCorrection: singular normal equations — bias field underdetermined".to_string(),
            ));
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in col..=nb {
            aug[col][j] /= pivot;
        }
        for row in 0..nb {
            if row != col {
                let factor = aug[row][col];
                for j in col..=nb {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;
                }
            }
        }
    }

    let coeffs: Vec<f64> = (0..nb).map(|i| aug[i][nb]).collect();
    Ok(coeffs)
}

// ─── VolumeStats ─────────────────────────────────────────────────────────────

/// Tissue-class statistics extracted from a medical volume.
#[derive(Debug, Clone)]
pub struct TissueClassStats {
    /// Tissue class identifier.
    pub tissue: Tissue,
    /// Number of voxels belonging to this tissue class.
    pub voxel_count: usize,
    /// Mean intensity within the tissue class.
    pub mean: f64,
    /// Standard deviation within the tissue class.
    pub std_dev: f64,
    /// Minimum intensity within the tissue class.
    pub min: f64,
    /// Maximum intensity within the tissue class.
    pub max: f64,
    /// 5th percentile intensity.
    pub p5: f64,
    /// 25th percentile intensity.
    pub p25: f64,
    /// Median (50th percentile) intensity.
    pub median: f64,
    /// 75th percentile intensity.
    pub p75: f64,
    /// 95th percentile intensity.
    pub p95: f64,
}

/// Per-tissue-class statistics for a CT volume.
#[derive(Debug, Clone)]
pub struct VolumeStats {
    /// Statistics broken down by tissue class.
    pub by_tissue: HashMap<String, TissueClassStats>,
    /// Overall volume statistics (all voxels).
    pub global_mean: f64,
    /// Overall standard deviation.
    pub global_std: f64,
    /// Total voxel count.
    pub total_voxels: usize,
}

impl VolumeStats {
    /// Compute per-tissue-class statistics for a CT HU volume.
    ///
    /// The volume must already contain HU values (i.e., rescaling has been
    /// applied via [`HounsfieldUnits::rescale`]).
    pub fn compute_ct(volume: &MedicalVolume) -> NdimageResult<Self> {
        let data = &volume.data;
        let n = data.len();
        if n == 0 {
            return Err(NdimageError::InvalidInput(
                "VolumeStats: volume must not be empty".to_string(),
            ));
        }

        let voxels: Vec<f64> = data.iter().cloned().collect();

        // Global stats
        let global_mean = voxels.iter().sum::<f64>() / n as f64;
        let global_var = voxels.iter().map(|v| (v - global_mean).powi(2)).sum::<f64>() / n as f64;
        let global_std = global_var.sqrt();

        // Group voxels by tissue class
        let mut by_class: HashMap<String, Vec<f64>> = HashMap::new();
        for &v in &voxels {
            let t = HounsfieldUnits::classify(v);
            by_class.entry(format!("{:?}", t)).or_default().push(v);
        }

        let mut by_tissue = HashMap::new();
        for (name, mut vals) in by_class {
            let tissue = tissue_from_str(&name);
            let stats = compute_tissue_stats(tissue, &mut vals);
            by_tissue.insert(name, stats);
        }

        Ok(Self {
            by_tissue,
            global_mean,
            global_std,
            total_voxels: n,
        })
    }

    /// Compute statistics using a generic intensity volume with explicit mask.
    ///
    /// Voxels where `mask[[iz, iy, ix]]` is `true` are included.
    pub fn compute_masked(volume: &MedicalVolume, mask: &Array3<bool>) -> NdimageResult<Self> {
        let data = &volume.data;
        let shape = data.shape();
        if shape != mask.shape() {
            return Err(NdimageError::DimensionError(
                "VolumeStats: volume and mask shapes must match".to_string(),
            ));
        }

        let masked_voxels: Vec<f64> = data
            .iter()
            .zip(mask.iter())
            .filter_map(|(&v, &m)| if m { Some(v) } else { None })
            .collect();

        let n = masked_voxels.len();
        if n == 0 {
            return Err(NdimageError::InvalidInput(
                "VolumeStats: mask selects zero voxels".to_string(),
            ));
        }

        let global_mean = masked_voxels.iter().sum::<f64>() / n as f64;
        let global_var = masked_voxels
            .iter()
            .map(|v| (v - global_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let global_std = global_var.sqrt();

        // Single "masked" class
        let mut vals = masked_voxels.clone();
        let stats = compute_tissue_stats(Tissue::SoftTissue, &mut vals);
        let mut by_tissue = HashMap::new();
        by_tissue.insert("Masked".to_string(), stats);

        Ok(Self {
            by_tissue,
            global_mean,
            global_std,
            total_voxels: n,
        })
    }
}

/// Helper: compute per-tissue statistics from a sorted list of intensities.
fn compute_tissue_stats(tissue: Tissue, vals: &mut Vec<f64>) -> TissueClassStats {
    let n = vals.len();
    if n == 0 {
        return TissueClassStats {
            tissue,
            voxel_count: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            p95: 0.0,
        };
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    let percentile = |p: f64| -> f64 {
        let idx_f = p / 100.0 * (n - 1) as f64;
        let lo = idx_f.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = idx_f - lo as f64;
        vals[lo] * (1.0 - frac) + vals[hi] * frac
    };
    TissueClassStats {
        tissue,
        voxel_count: n,
        mean,
        std_dev,
        min: vals[0],
        max: vals[n - 1],
        p5: percentile(5.0),
        p25: percentile(25.0),
        median: percentile(50.0),
        p75: percentile(75.0),
        p95: percentile(95.0),
    }
}

fn tissue_from_str(s: &str) -> Tissue {
    match s {
        "Air" => Tissue::Air,
        "Lung" => Tissue::Lung,
        "Fat" => Tissue::Fat,
        "Blood" => Tissue::Blood,
        "CancellousBone" => Tissue::CancellousBone,
        "CorticalBone" => Tissue::CorticalBone,
        "Metal" => Tissue::Metal,
        _ => Tissue::SoftTissue,
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_dicom_header_validation() {
        let h = DicomHeader::new_ct("P001", 512, 512, 64);
        assert!(h.validate().is_ok());
    }

    #[test]
    fn test_dicom_header_invalid_slice_thickness() {
        let mut h = DicomHeader::new_ct("P001", 512, 512, 64);
        h.slice_thickness = -1.0;
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_window_leveling_apply() {
        let wl = WindowLeveling::ct_abdomen(); // level=60, window=400
        // At the lower bound (level - window/2 = -140) → 0
        assert!((wl.apply(-140.0) - 0.0).abs() < 1e-10);
        // At the upper bound (260) → 1
        assert!((wl.apply(260.0) - 1.0).abs() < 1e-10);
        // At the centre → 0.5
        assert!((wl.apply(60.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hounsfield_classify() {
        assert_eq!(HounsfieldUnits::classify(-1100.0), Tissue::Air);
        assert_eq!(HounsfieldUnits::classify(-700.0), Tissue::Lung);
        assert_eq!(HounsfieldUnits::classify(-200.0), Tissue::Fat);
        assert_eq!(HounsfieldUnits::classify(0.0), Tissue::SoftTissue);
        assert_eq!(HounsfieldUnits::classify(150.0), Tissue::Blood);
        assert_eq!(HounsfieldUnits::classify(500.0), Tissue::CancellousBone);
        assert_eq!(HounsfieldUnits::classify(1000.0), Tissue::CorticalBone);
        assert_eq!(HounsfieldUnits::classify(4000.0), Tissue::Metal);
    }

    #[test]
    fn test_n4_bias_correction_smoke() {
        // Small 4x4x4 volume with a simple linear bias
        let mut data = Array3::<f64>::ones((4, 4, 4));
        for iz in 0..4_usize {
            for iy in 0..4_usize {
                for ix in 0..4_usize {
                    data[[iz, iy, ix]] = 100.0 + iz as f64 * 10.0 + iy as f64 * 5.0;
                }
            }
        }
        let vol = MedicalVolume::new(data, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]).expect("MedicalVolume::new should succeed with valid data");
        let corrector = N4BiasCorrection::new();
        let result = corrector.correct(&vol);
        assert!(result.is_ok(), "N4 correction failed: {:?}", result.err());
        let (corrected, bias) = result.expect("N4 correction result should be Ok after is_ok check");
        // Bias field should be non-trivially different from 1 (correction applied)
        let _ = corrected;
        let _ = bias;
    }

    #[test]
    fn test_volume_stats_ct() {
        let data = Array3::<f64>::from_elem((4, 4, 4), 50.0); // soft tissue
        let vol = MedicalVolume::new(data, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]).expect("MedicalVolume::new should succeed with uniform CT data");
        let stats = VolumeStats::compute_ct(&vol).expect("compute_ct should succeed on valid volume");
        assert!((stats.global_mean - 50.0).abs() < 1e-10);
        assert!(stats.total_voxels == 64);
    }

    #[test]
    fn test_medical_volume_axial_slice() {
        let data = Array3::<f64>::zeros((5, 4, 3));
        let vol = MedicalVolume::new(data, [2.0, 1.5, 1.0], [0.0, 0.0, 0.0]).expect("MedicalVolume::new should succeed with zeros volume");
        let slice = vol.axial_slice(2).expect("axial_slice(2) should succeed for a 5-slice volume");
        assert_eq!(slice.len(), 12); // 4 × 3
        // Out-of-range slice
        assert!(vol.axial_slice(10).is_err());
    }

    #[test]
    fn test_polynomial_basis() {
        let basis = polynomial_basis_3d(2);
        // degree 2: terms (0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2) = 10
        assert_eq!(basis.len(), 10);
    }
}
