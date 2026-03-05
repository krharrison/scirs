//! Spatial ICA for neuroimaging and group-level analysis
//!
//! Spatial ICA (sICA) differs from temporal ICA in that spatial patterns (maps)
//! are treated as the statistically independent sources.  The data matrix is
//! transposed so that:
//!
//! - rows  = voxels  (N_v)
//! - cols  = time-points (T)
//!
//! and FastICA is run to find spatially-independent component maps.  The mixing
//! matrix then describes how these spatial patterns combine across time.
//!
//! ## Group ICA
//!
//! For multi-subject studies a two-stage approach is used:
//!
//! 1. Concatenate subjects along the time axis.
//! 2. Perform a single sICA on the concatenated data.
//! 3. The resulting spatial maps are shared; per-subject time courses are
//!    recovered by projecting each subject's data onto the spatial maps.
//!
//! ## ICASSO stability analysis
//!
//! ICASSO (Himberg et al., 2004) runs ICA many times with perturbed initialisations
//! (random seeds) and clusters the resulting components to identify "stable" ICs.
//!
//! # References
//!
//! - McKeown, M.J. et al. (1998). Analysis of fMRI data by blind separation
//!   into independent spatial components. *Human Brain Mapping*.
//! - Calhoun, V.D. et al. (2001). A method for making group inferences from
//!   functional MRI data using independent component analysis. *Human Brain Mapping*.
//! - Himberg, J. et al. (2004). Validating the independent components of
//!   neuroimaging time series via clustering and visualization. *NeuroImage*.

use crate::error::{Result, TransformError};
use crate::reduction::fastica::{FastICA, IcaAlgorithm, NonLinearity};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::svd;

const EPSILON: f64 = 1e-12;

// ─── Core data structures ────────────────────────────────────────────────────

/// Result of a fitted Spatial ICA model.
///
/// Given input data of shape `(T, V)` (time × voxels):
/// - `sources`   has shape `(V, C)` — C spatially-independent component maps.
/// - `mixing`    has shape `(T, C)` — time courses (how components mix over time).
/// - `whitening` has shape `(C, T)` — the whitening matrix applied to the
///   transposed data before the FastICA rotation.
#[derive(Debug, Clone)]
pub struct SpatialIcaModel {
    /// Spatial component maps, shape `(n_voxels, n_components)`.
    pub sources: Array2<f64>,
    /// Time-course mixing matrix, shape `(n_time, n_components)`.
    pub mixing: Array2<f64>,
    /// Whitening matrix used during fitting, shape `(n_components, n_time)`.
    pub whitening: Array2<f64>,
    /// Number of independent components.
    pub n_components: usize,
}

/// Configuration for the Spatial ICA algorithm.
#[derive(Debug, Clone)]
pub struct SpatialIca {
    /// Number of independent components to extract.
    pub n_components: usize,
    /// Maximum FastICA iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Non-linearity used by FastICA.
    pub non_linearity: NonLinearity,
    /// FastICA variant (deflation or symmetric).
    pub algorithm: IcaAlgorithm,
}

impl Default for SpatialIca {
    fn default() -> Self {
        Self {
            n_components: 20,
            max_iter: 200,
            tol: 1e-4,
            non_linearity: NonLinearity::LogCosh,
            algorithm: IcaAlgorithm::Symmetric,
        }
    }
}

impl SpatialIca {
    /// Create a new `SpatialIca` with `n_components` components and default settings.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// Override the maximum number of FastICA iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Override the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-15);
        self
    }

    /// Override the non-linearity function.
    pub fn with_non_linearity(mut self, nl: NonLinearity) -> Self {
        self.non_linearity = nl;
        self
    }

    /// Override the FastICA algorithm variant.
    pub fn with_algorithm(mut self, alg: IcaAlgorithm) -> Self {
        self.algorithm = alg;
        self
    }

    /// Fit Spatial ICA to neuroimaging data.
    ///
    /// # Arguments
    /// * `data` — 2-D array of shape `(n_time, n_voxels)`.
    ///
    /// # Returns
    /// A fitted [`SpatialIcaModel`].
    ///
    /// # Errors
    /// Returns an error if the data dimensions are incompatible with the
    /// requested number of components.
    pub fn fit(&self, data: &Array2<f64>) -> Result<SpatialIcaModel> {
        let (n_time, n_voxels) = (data.shape()[0], data.shape()[1]);

        if n_time < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 time-points are required for Spatial ICA".to_string(),
            ));
        }
        if n_voxels < self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "n_voxels ({n_voxels}) must be >= n_components ({})",
                self.n_components
            )));
        }
        if n_time < self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "n_time ({n_time}) must be >= n_components ({})",
                self.n_components
            )));
        }

        // Transpose: rows = voxels, cols = time
        // FastICA treats rows as "samples", so voxels become samples
        let x_t = data.t().to_owned(); // shape (V, T)

        // Run standard FastICA on the transposed data
        let mut ica = FastICA::new(self.n_components)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_non_linearity(self.non_linearity)
            .with_algorithm(self.algorithm)
            .with_whiten(true);

        ica.fit(&x_t)?;

        // sources: (V, C) — the spatially-independent component maps
        let sources = ica.transform(&x_t)?;

        // Retrieve whitening matrix (C, T)
        let whitening = ica
            .whitening_matrix()
            .ok_or_else(|| {
                TransformError::ComputationError(
                    "Whitening matrix not available after fit".to_string(),
                )
            })?
            .clone();

        // Time courses: project original data onto the spatial maps
        // mixing = data @ sources @ (sources.T @ sources)^{-1}  — shape (T, C)
        let mixing = compute_time_courses(data, &sources)?;

        Ok(SpatialIcaModel {
            sources,
            mixing,
            whitening,
            n_components: self.n_components,
        })
    }
}

// ─── Group ICA ───────────────────────────────────────────────────────────────

/// Perform group-level Spatial ICA.
///
/// All subjects' data are concatenated along the **time** axis before running
/// a single sICA.  The resulting spatial maps are common to the group; each
/// subject's time courses are recovered via least-squares projection.
///
/// # Arguments
/// * `subjects` — slice of `(n_time_i, n_voxels)` arrays, one per subject.
///   All subjects must have the **same** number of voxels.
/// * `n_components` — number of independent components.
///
/// # Returns
/// A [`SpatialIcaModel`] fitted on the concatenated group data.
pub fn group_ica(subjects: &[Array2<f64>], n_components: usize) -> Result<SpatialIcaModel> {
    if subjects.is_empty() {
        return Err(TransformError::InvalidInput(
            "subjects slice must not be empty".to_string(),
        ));
    }

    let n_voxels = subjects[0].shape()[1];
    for (idx, s) in subjects.iter().enumerate() {
        if s.shape()[1] != n_voxels {
            return Err(TransformError::InvalidInput(format!(
                "Subject {idx} has {} voxels but expected {n_voxels}",
                s.shape()[1]
            )));
        }
        if s.shape()[0] < 2 {
            return Err(TransformError::InvalidInput(format!(
                "Subject {idx} has fewer than 2 time-points"
            )));
        }
    }

    // Concatenate along axis 0 (time)
    let total_time: usize = subjects.iter().map(|s| s.shape()[0]).sum();
    let mut concatenated = Array2::<f64>::zeros((total_time, n_voxels));
    let mut row = 0;
    for s in subjects {
        let t = s.shape()[0];
        for tr in 0..t {
            for v in 0..n_voxels {
                concatenated[[row + tr, v]] = s[[tr, v]];
            }
        }
        row += t;
    }

    let sica = SpatialIca::new(n_components);
    sica.fit(&concatenated)
}

// ─── ICASSO stability analysis ────────────────────────────────────────────────

/// ICASSO-like stability analysis for ICA components.
///
/// Runs ICA `n_runs` times with different (deterministic) random initialisations,
/// clusters the resulting components by mutual correlation, and returns:
///
/// * `components` — stable component maps, shape `(n_voxels, n_components)`.
/// * `stability`  — per-component stability index ∈ [0, 1] (mean intra-cluster
///   absolute correlation minus mean inter-cluster absolute correlation).
///
/// # Arguments
/// * `data` — `(n_time, n_voxels)` array.
/// * `n_runs` — number of ICA repetitions (≥ 2).
/// * `n_components` — number of ICs to extract each run.
///
/// # References
/// - Himberg, J. et al. (2004). *NeuroImage*, 22(3), 1214-1222.
pub fn icasso(
    data: &Array2<f64>,
    n_runs: usize,
    n_components: usize,
) -> Result<(Array2<f64>, Vec<f64>)> {
    if n_runs < 2 {
        return Err(TransformError::InvalidInput(
            "ICASSO requires at least 2 ICA runs".to_string(),
        ));
    }

    let (n_time, n_voxels) = (data.shape()[0], data.shape()[1]);

    if n_voxels < n_components || n_time < n_components {
        return Err(TransformError::InvalidInput(format!(
            "n_components ({n_components}) must be <= min(n_time={n_time}, n_voxels={n_voxels})"
        )));
    }

    // Collect all components from all runs:  shape (n_runs * C, V)
    let total = n_runs * n_components;
    let mut all_components = Array2::<f64>::zeros((total, n_voxels));

    for run in 0..n_runs {
        // Vary tolerance slightly to encourage different local optima
        let tol = 1e-4 * (1.0 + run as f64 * 0.1);
        let sica = SpatialIca::new(n_components)
            .with_max_iter(500)
            .with_tol(tol)
            .with_non_linearity(NonLinearity::LogCosh)
            .with_algorithm(IcaAlgorithm::Symmetric);

        let model = sica.fit(data)?;

        // Normalise each component to unit norm before clustering
        for c in 0..n_components {
            let mut norm_sq = 0.0_f64;
            for v in 0..n_voxels {
                let val = model.sources[[v, c]];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt().max(EPSILON);
            let idx = run * n_components + c;
            for v in 0..n_voxels {
                all_components[[idx, v]] = model.sources[[v, c]] / norm;
            }
        }
    }

    // Build absolute-correlation similarity matrix  S[i,j] = |corr(i,j)|
    let similarity = build_abs_correlation_matrix(&all_components)?;

    // Cluster by greedy best-match across runs
    let assignments = greedy_cluster(&similarity, n_runs, n_components)?;

    // For each cluster, compute the mean component (centroid) and stability index
    let mut stable_components = Array2::<f64>::zeros((n_voxels, n_components));
    let mut stability = vec![0.0_f64; n_components];

    for (cluster_id, members) in assignments.iter().enumerate() {
        if members.is_empty() {
            continue;
        }

        // Centroid: mean of all members (after sign alignment to first member)
        let mut centroid = Array1::<f64>::zeros(n_voxels);
        let ref_row: Vec<f64> = (0..n_voxels)
            .map(|v| all_components[[members[0], v]])
            .collect();

        for &idx in members {
            let dot: f64 = (0..n_voxels)
                .map(|v| ref_row[v] * all_components[[idx, v]])
                .sum();
            let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
            for v in 0..n_voxels {
                centroid[v] += sign * all_components[[idx, v]];
            }
        }
        let n_members = members.len() as f64;
        centroid.mapv_inplace(|v| v / n_members);

        for v in 0..n_voxels {
            stable_components[[v, cluster_id]] = centroid[v];
        }

        // Stability = mean intra-cluster |corr| – mean inter-cluster |corr|
        let intra = mean_within_cluster_similarity(&similarity, members);
        let inter = mean_between_cluster_similarity(&similarity, members, total);
        stability[cluster_id] = (intra - inter).clamp(0.0, 1.0);
    }

    Ok((stable_components, stability))
}

// ─── Helper: time-course projection ──────────────────────────────────────────

/// Compute time courses by least-squares projection of `data` onto `spatial_maps`.
///
/// Solves: `data ≈ mixing @ spatial_maps.T`  →
/// `mixing = data @ spatial_maps @ (spatial_maps.T @ spatial_maps)^{-1}`
///
/// # Arguments
/// * `data`         — `(T, V)` data matrix.
/// * `spatial_maps` — `(V, C)` component maps.
fn compute_time_courses(data: &Array2<f64>, spatial_maps: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_time, n_voxels) = (data.shape()[0], data.shape()[1]);
    let n_comp = spatial_maps.shape()[1];

    if spatial_maps.shape()[0] != n_voxels {
        return Err(TransformError::InvalidInput(format!(
            "spatial_maps rows ({}) must equal data columns ({n_voxels})",
            spatial_maps.shape()[0]
        )));
    }

    // A = S^T S  (C × C)
    let mut a = Array2::<f64>::zeros((n_comp, n_comp));
    for i in 0..n_comp {
        for j in 0..n_comp {
            let mut dot = 0.0_f64;
            for v in 0..n_voxels {
                dot += spatial_maps[[v, i]] * spatial_maps[[v, j]];
            }
            a[[i, j]] = dot;
        }
    }

    // b = data @ spatial_maps  (T × C)
    let mut b = Array2::<f64>::zeros((n_time, n_comp));
    for t in 0..n_time {
        for c in 0..n_comp {
            let mut dot = 0.0_f64;
            for v in 0..n_voxels {
                dot += data[[t, v]] * spatial_maps[[v, c]];
            }
            b[[t, c]] = dot;
        }
    }

    // Solve A x = b^T  via pseudo-inverse of A
    let a_pinv = pinv_small(&a)?;

    // mixing = b @ a_pinv   (T × C)
    let mut mixing = Array2::<f64>::zeros((n_time, n_comp));
    for t in 0..n_time {
        for c in 0..n_comp {
            let mut dot = 0.0_f64;
            for k in 0..n_comp {
                dot += b[[t, k]] * a_pinv[[k, c]];
            }
            mixing[[t, c]] = dot;
        }
    }

    Ok(mixing)
}

// ─── Helper: absolute-correlation matrix ─────────────────────────────────────

fn build_abs_correlation_matrix(components: &Array2<f64>) -> Result<Array2<f64>> {
    let n = components.shape()[0];
    let d = components.shape()[1];
    let mut sim = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        sim[[i, i]] = 1.0;
        for j in (i + 1)..n {
            let mut dot = 0.0_f64;
            let mut norm_i = 0.0_f64;
            let mut norm_j = 0.0_f64;
            for k in 0..d {
                let ai = components[[i, k]];
                let aj = components[[j, k]];
                dot += ai * aj;
                norm_i += ai * ai;
                norm_j += aj * aj;
            }
            let denom = (norm_i * norm_j).sqrt().max(EPSILON);
            let corr = (dot / denom).abs();
            sim[[i, j]] = corr;
            sim[[j, i]] = corr;
        }
    }
    Ok(sim)
}

// ─── Helper: greedy clustering ────────────────────────────────────────────────

/// Assign the `n_runs × n_components` component indices into `n_components`
/// clusters using a greedy nearest-neighbor approach.
///
/// Strategy: for each cluster seed (the c-th component of run 0), find the
/// best matching component from each subsequent run (maximise similarity).
fn greedy_cluster(
    similarity: &Array2<f64>,
    n_runs: usize,
    n_components: usize,
) -> Result<Vec<Vec<usize>>> {
    let mut clusters: Vec<Vec<usize>> = (0..n_components).map(|c| vec![c]).collect();

    for run in 1..n_runs {
        let start = run * n_components;
        let mut used = vec![false; n_components];

        // Build a score matrix: cluster c vs candidate run_c
        let mut scores = Array2::<f64>::zeros((n_components, n_components));
        for c in 0..n_components {
            for run_c in 0..n_components {
                let cand = start + run_c;
                let members = &clusters[c];
                let mean_sim: f64 = members
                    .iter()
                    .map(|&m| similarity[[m, cand]])
                    .sum::<f64>()
                    / members.len() as f64;
                scores[[c, run_c]] = mean_sim;
            }
        }

        // Greedy assignment: pick highest score first
        for _ in 0..n_components {
            let mut best_score = -1.0_f64;
            let mut best_c = 0;
            let mut best_run_c = 0;

            for c in 0..n_components {
                if clusters[c].len() > run {
                    continue; // already assigned for this run
                }
                for run_c in 0..n_components {
                    if used[run_c] {
                        continue;
                    }
                    if scores[[c, run_c]] > best_score {
                        best_score = scores[[c, run_c]];
                        best_c = c;
                        best_run_c = run_c;
                    }
                }
            }

            if best_score < 0.0 {
                break;
            }
            clusters[best_c].push(start + best_run_c);
            used[best_run_c] = true;
        }
    }

    Ok(clusters)
}

fn mean_within_cluster_similarity(similarity: &Array2<f64>, members: &[usize]) -> f64 {
    let n = members.len();
    if n <= 1 {
        return 1.0;
    }
    let mut total = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += similarity[[members[i], members[j]]];
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

fn mean_between_cluster_similarity(
    similarity: &Array2<f64>,
    members: &[usize],
    total: usize,
) -> f64 {
    let member_set: std::collections::HashSet<usize> = members.iter().copied().collect();
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for &m in members {
        for j in 0..total {
            if !member_set.contains(&j) {
                sum += similarity[[m, j]];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

// ─── Pseudo-inverse of a small square matrix via SVD ─────────────────────────

fn pinv_small(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.shape()[0];
    let (u, s, vt) = svd::<f64>(&a.view(), true, None).map_err(TransformError::LinalgError)?;

    // Compute pinv = V @ diag(1/s) @ U^T
    let mut pinv = Array2::<f64>::zeros((n, n));
    let threshold = EPSILON * s[0].abs().max(EPSILON);

    for i in 0..n {
        if s[i].abs() > threshold {
            let inv_s = 1.0 / s[i];
            for r in 0..n {
                for c in 0..n {
                    // vt is (n, n), u is (n, n)
                    pinv[[r, c]] += vt[[i, r]] * inv_s * u[[c, i]];
                }
            }
        }
    }
    Ok(pinv)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_fmri(n_time: usize, n_voxels: usize, n_sources: usize) -> Array2<f64> {
        // Generate n_sources independent spatial maps and mix them
        let mut sources = Array2::<f64>::zeros((n_voxels, n_sources));
        for s in 0..n_sources {
            for v in 0..n_voxels {
                let val = ((v as f64 + s as f64 * 10.0) * 0.3).sin();
                sources[[v, s]] = val;
            }
        }
        // Time courses
        let mut time_courses = Array2::<f64>::zeros((n_time, n_sources));
        for t in 0..n_time {
            for s in 0..n_sources {
                time_courses[[t, s]] = ((t as f64 + s as f64 * 7.0) * 0.2).sin();
            }
        }
        // data = time_courses @ sources.T
        let mut data = Array2::<f64>::zeros((n_time, n_voxels));
        for t in 0..n_time {
            for v in 0..n_voxels {
                let mut val = 0.0;
                for s in 0..n_sources {
                    val += time_courses[[t, s]] * sources[[v, s]];
                }
                data[[t, v]] = val;
            }
        }
        data
    }

    #[test]
    fn test_spatial_ica_shapes() {
        let data = make_synthetic_fmri(50, 100, 3);
        let sica = SpatialIca::new(3).with_max_iter(100);
        let model = sica.fit(&data).expect("SpatialIca::fit failed");

        assert_eq!(model.sources.shape(), &[100, 3]);
        assert_eq!(model.mixing.shape(), &[50, 3]);
        assert_eq!(model.n_components, 3);
    }

    #[test]
    fn test_group_ica_shapes() {
        let s1 = make_synthetic_fmri(40, 80, 3);
        let s2 = make_synthetic_fmri(30, 80, 3);
        let model = group_ica(&[s1, s2], 3).expect("group_ica failed");

        assert_eq!(model.sources.shape(), &[80, 3]);
        assert_eq!(model.mixing.shape(), &[70, 3]); // 40+30 time points
        assert_eq!(model.n_components, 3);
    }

    #[test]
    fn test_icasso_stability() {
        let data = make_synthetic_fmri(60, 80, 3);
        let (components, stability) = icasso(&data, 3, 3).expect("icasso failed");

        assert_eq!(components.shape(), &[80, 3]);
        assert_eq!(stability.len(), 3);
        for &s in &stability {
            assert!(s >= 0.0 && s <= 1.0, "stability out of [0,1]: {s}");
        }
    }

    #[test]
    fn test_spatial_ica_invalid_input() {
        // Too few time-points
        let data = Array2::<f64>::zeros((1, 100));
        let sica = SpatialIca::new(3);
        assert!(sica.fit(&data).is_err());

        // n_components > n_time
        let data2 = Array2::<f64>::zeros((5, 100));
        let sica2 = SpatialIca::new(10);
        assert!(sica2.fit(&data2).is_err());
    }
}
