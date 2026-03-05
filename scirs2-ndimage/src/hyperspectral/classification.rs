//! Hyperspectral Image Classification
//!
//! Pixel-wise classification methods based on spectral similarity measures:
//! - Spectral Angle Mapper (SAM)
//! - Spectral Information Divergence (SID)
//! - Spectral Correlation Mapper (SCM)
//! - Matched Subspace Detector (MSD)
//! - Hard classification from abundance maps

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{NdimageError, NdimageResult};
use crate::hyperspectral::unmixing::HyperspectralImage;

// ─────────────────────────────────────────────────────────────────────────────
// Common output type
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel classification result.
#[derive(Debug, Clone)]
pub struct ClassificationMap {
    /// Class label for each pixel (0-based index into the reference library).
    /// Value = `n_classes` means "unclassified" (score exceeds threshold).
    pub labels: Vec<usize>,
    /// Best-match score per pixel (lower = better for distance metrics,
    /// higher = better for correlation metrics).
    pub scores: Vec<f64>,
    /// Number of reference classes.
    pub n_classes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean norm of a 1-D slice.
#[inline]
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Mean of a slice.
#[inline]
fn mean_slice(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Clamp to `[0, 1]` avoiding log(0).
#[inline]
fn safe_prob(v: f64) -> f64 {
    v.max(1e-300).min(1.0 - 1e-300)
}

/// Normalise a spectrum to a probability distribution (sum to 1, all positive).
fn to_prob_dist(spectrum: &[f64]) -> Vec<f64> {
    let min_val = spectrum.iter().cloned().fold(f64::INFINITY, f64::min);
    let shifted: Vec<f64> = spectrum.iter().map(|&x| x - min_val + 1e-10).collect();
    let total: f64 = shifted.iter().sum();
    shifted.iter().map(|&x| x / total.max(1e-300)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral Angle Mapper (SAM)
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral Angle Mapper (SAM) classifier.
///
/// Assigns each pixel to the reference spectrum with the smallest spectral
/// angle.  The spectral angle is invariant to illumination changes (it is
/// a dot-product cosine measure):
/// `θ = arccos( <x, r> / (‖x‖ ‖r‖) )`
///
/// # Arguments
/// * `image`       - Hyperspectral image `[N_pixels, N_bands]`.
/// * `references`  - Reference spectral library `[N_classes, N_bands]`.
/// * `threshold`   - Maximum spectral angle (radians) to classify a pixel.
///                   Pixels exceeding this are labelled `n_classes` (unclassified).
///                   Use `f64::MAX` to classify all pixels.
///
/// # Returns
/// [`ClassificationMap`] with per-pixel labels (0-based) and angles.
pub fn sam_classifier(
    image: &HyperspectralImage,
    references: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<ClassificationMap> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let n_classes = references.nrows();

    if n_classes == 0 {
        return Err(NdimageError::InvalidInput("references must have at least one row".into()));
    }
    if references.ncols() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "references.ncols() {} != n_bands {}",
            references.ncols(), n_bands
        )));
    }

    // Pre-compute norms of reference spectra.
    let ref_norms: Vec<f64> = (0..n_classes)
        .map(|c| norm(references.row(c).to_slice().unwrap_or(&[])))
        .collect();

    let mut labels = vec![n_classes; n_pixels]; // unclassified by default.
    let mut scores = vec![f64::MAX; n_pixels];

    for p in 0..n_pixels {
        let pixel = image.data.row(p);
        let px_slice = pixel.to_slice().unwrap_or(&[]);
        let px_norm = norm(px_slice);

        if px_norm < 1e-14 {
            continue; // Zero pixel — leave unclassified.
        }

        for c in 0..n_classes {
            let ref_slice = references.row(c).to_slice().unwrap_or(&[]);
            if ref_norms[c] < 1e-14 {
                continue;
            }
            let cos_theta = (dot(px_slice, ref_slice) / (px_norm * ref_norms[c])).clamp(-1.0, 1.0);
            let angle = cos_theta.acos();
            if angle < scores[p] {
                scores[p] = angle;
                labels[p] = c;
            }
        }

        if scores[p] > threshold {
            labels[p] = n_classes; // Mark as unclassified.
        }
    }

    Ok(ClassificationMap { labels, scores, n_classes })
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral Information Divergence (SID)
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral Information Divergence (SID) classifier.
///
/// Models each spectrum as a probability distribution and computes the
/// symmetric KL divergence between pixel and reference:
/// `SID(x, r) = KL(x||r) + KL(r||x)`
///
/// Smaller SID → more similar spectra.
///
/// # Arguments
/// * `image`       - Hyperspectral image.
/// * `references`  - Reference spectral library `[N_classes, N_bands]`.
/// * `threshold`   - Maximum SID value to classify a pixel.
///
/// # Returns
/// [`ClassificationMap`] with per-pixel labels and SID scores.
pub fn sid_classifier(
    image: &HyperspectralImage,
    references: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<ClassificationMap> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let n_classes = references.nrows();

    if n_classes == 0 {
        return Err(NdimageError::InvalidInput("references must have at least one row".into()));
    }
    if references.ncols() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "references.ncols() {} != n_bands {}",
            references.ncols(), n_bands
        )));
    }

    // Pre-normalise references to probability distributions.
    let ref_probs: Vec<Vec<f64>> = (0..n_classes)
        .map(|c| to_prob_dist(references.row(c).to_slice().unwrap_or(&[])))
        .collect();

    let mut labels = vec![n_classes; n_pixels];
    let mut scores = vec![f64::MAX; n_pixels];

    for p in 0..n_pixels {
        let pixel = image.data.row(p);
        let px_slice = pixel.to_slice().unwrap_or(&[]);
        let px_prob = to_prob_dist(px_slice);

        for c in 0..n_classes {
            let rp = &ref_probs[c];
            // KL(x || r)
            let kl_xr: f64 = (0..n_bands)
                .map(|b| {
                    let px = safe_prob(px_prob[b]);
                    let pr = safe_prob(rp[b]);
                    px * (px / pr).ln()
                })
                .sum();
            // KL(r || x)
            let kl_rx: f64 = (0..n_bands)
                .map(|b| {
                    let px = safe_prob(px_prob[b]);
                    let pr = safe_prob(rp[b]);
                    pr * (pr / px).ln()
                })
                .sum();
            let sid = kl_xr + kl_rx;
            if sid < scores[p] {
                scores[p] = sid;
                labels[p] = c;
            }
        }

        if scores[p] > threshold {
            labels[p] = n_classes;
        }
    }

    Ok(ClassificationMap { labels, scores, n_classes })
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral Correlation Mapper (SCM)
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral Correlation Mapper (SCM) — correlation-based spectral matching.
///
/// Computes the Pearson correlation coefficient between each pixel and each
/// reference spectrum.  Higher correlation → better match.
///
/// `SCM(x, r) = corr(x, r) ∈ [-1, 1]`
///
/// # Arguments
/// * `image`       - Hyperspectral image.
/// * `references`  - Reference spectral library `[N_classes, N_bands]`.
/// * `threshold`   - Minimum correlation to classify (default 0.0 accepts all).
///
/// # Returns
/// [`ClassificationMap`] where `scores` contains the best correlation value
/// (higher is better).
pub fn spectral_correlation_mapper(
    image: &HyperspectralImage,
    references: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<ClassificationMap> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let n_classes = references.nrows();

    if n_classes == 0 {
        return Err(NdimageError::InvalidInput("references must have at least one row".into()));
    }
    if references.ncols() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "references.ncols() {} != n_bands {}",
            references.ncols(), n_bands
        )));
    }

    // Pre-centre and pre-normalise reference spectra.
    let ref_stats: Vec<(Vec<f64>, f64)> = (0..n_classes)
        .map(|c| {
            let row = references.row(c);
            let rs = row.to_slice().unwrap_or(&[]);
            let m = mean_slice(rs);
            let centred: Vec<f64> = rs.iter().map(|&x| x - m).collect();
            let n = norm(&centred).max(1e-15);
            (centred, n)
        })
        .collect();

    let mut labels = vec![n_classes; n_pixels];
    let mut scores = vec![f64::NEG_INFINITY; n_pixels];

    for p in 0..n_pixels {
        let pixel = image.data.row(p);
        let px_slice = pixel.to_slice().unwrap_or(&[]);
        let px_mean = mean_slice(px_slice);
        let px_centred: Vec<f64> = px_slice.iter().map(|&x| x - px_mean).collect();
        let px_norm = norm(&px_centred);

        if px_norm < 1e-14 {
            continue;
        }

        for c in 0..n_classes {
            let (ref c_vec, ref c_norm) = ref_stats[c];
            let corr = dot(&px_centred, c_vec) / (px_norm * c_norm);
            if corr > scores[p] {
                scores[p] = corr;
                labels[p] = c;
            }
        }

        if scores[p] < threshold {
            labels[p] = n_classes;
        }
    }

    Ok(ClassificationMap { labels, scores, n_classes })
}

// ─────────────────────────────────────────────────────────────────────────────
// Matched Subspace Detector (MSD)
// ─────────────────────────────────────────────────────────────────────────────

/// Matched Subspace Detector (MSD) for hyperspectral target detection.
///
/// Projects each pixel onto the subspace spanned by the reference spectra
/// (a.k.a. the signal subspace) and computes the energy in that projection.
/// High projection energy → pixel is "matched" by the subspace.
///
/// More precisely, for a subspace basis `U` (orthonormal columns):
/// `score(x) = ‖U Uᵀ x‖² / ‖x‖²`
///
/// # Arguments
/// * `image`       - Hyperspectral image.
/// * `subspace`    - Subspace basis vectors as columns `[N_bands, K]` (need not be
///                   orthonormal — the function will orthonormalise internally).
/// * `threshold`   - Minimum score `∈ [0, 1]` to flag a match.
///
/// # Returns
/// `(detection_flags [N_pixels], scores [N_pixels])`.
pub fn subspace_detector(
    image: &HyperspectralImage,
    subspace: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<(Vec<bool>, Vec<f64>)> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let k = subspace.ncols();

    if subspace.nrows() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "subspace.nrows() {} != n_bands {}",
            subspace.nrows(), n_bands
        )));
    }
    if k == 0 {
        return Err(NdimageError::InvalidInput("subspace must have at least one column".into()));
    }
    if !(0.0..=1.0).contains(&threshold) {
        return Err(NdimageError::InvalidInput("threshold must be in [0, 1]".into()));
    }

    // Orthonormalise the subspace via Gram–Schmidt.
    let mut orth = Array2::<f64>::zeros((n_bands, k));
    for j in 0..k {
        let mut col: Array1<f64> = subspace.column(j).to_owned();
        for i in 0..j {
            let qi = orth.column(i).to_owned();
            let proj: f64 = col.iter().zip(qi.iter()).map(|(a, b)| a * b).sum();
            col = col - qi * proj;
        }
        let col_norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if col_norm > 1e-14 {
            for i in 0..n_bands {
                orth[[i, j]] = col[i] / col_norm;
            }
        }
    }

    let mut detections = vec![false; n_pixels];
    let mut scores = vec![0.0_f64; n_pixels];

    for p in 0..n_pixels {
        let pixel = image.data.row(p);
        let px_slice = pixel.to_slice().unwrap_or(&[]);
        let px_norm_sq: f64 = px_slice.iter().map(|x| x * x).sum();

        if px_norm_sq < 1e-28 {
            continue;
        }

        // Projection energy: ‖Uᵀ x‖² = sum over columns of U of (uᵢᵀ x)².
        let proj_energy_sq: f64 = (0..k)
            .map(|j| {
                let qi = orth.column(j);
                let qi_slice = qi.to_slice().unwrap_or(&[]);
                let d = dot(px_slice, qi_slice);
                d * d
            })
            .sum();

        let score = proj_energy_sq / px_norm_sq;
        scores[p] = score;
        detections[p] = score >= threshold;
    }

    Ok((detections, scores))
}

// ─────────────────────────────────────────────────────────────────────────────
// Hard classification from abundance maps
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an abundance map to a hard classification (winner-takes-all).
///
/// Each pixel is assigned to the endmember with the highest abundance.
/// Optionally, pixels where no single abundance exceeds `min_abundance` are
/// labelled as unclassified (= `n_endmembers`).
///
/// # Arguments
/// * `abundances`    - Abundance matrix `[N_pixels, p]` (e.g., from FCLS).
/// * `min_abundance` - Minimum winning abundance to classify; use `0.0` to always classify.
///
/// # Returns
/// `ClassificationMap` where each pixel is labelled by its dominant endmember.
pub fn abundance_map_to_class(
    abundances: &Array2<f64>,
    min_abundance: f64,
) -> NdimageResult<ClassificationMap> {
    let n_pixels = abundances.nrows();
    let p = abundances.ncols();

    if p == 0 {
        return Err(NdimageError::InvalidInput("abundance matrix must have at least one column".into()));
    }

    let mut labels = vec![p; n_pixels]; // `p` = unclassified sentinel.
    let mut scores = vec![0.0_f64; n_pixels];

    for i in 0..n_pixels {
        let mut best_class = p;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..p {
            if abundances[[i, j]] > best_val {
                best_val = abundances[[i, j]];
                best_class = j;
            }
        }
        scores[i] = best_val;
        labels[i] = if best_val >= min_abundance { best_class } else { p };
    }

    Ok(ClassificationMap { labels, scores, n_classes: p })
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended SAM-SID hybrid (SAM × SID)
// ─────────────────────────────────────────────────────────────────────────────

/// SAM–SID hybrid classifier (SAMID).
///
/// Combines spectral angle and divergence into a single score:
/// `SAMID(x, r) = SAM(x, r) × SID(x, r)`
///
/// This captures both shape similarity (SAM) and distribution similarity (SID).
///
/// # Arguments
/// * `image`      - Hyperspectral image.
/// * `references` - `[N_classes, N_bands]`.
/// * `threshold`  - Maximum SAMID value for classification.
///
/// # Returns
/// [`ClassificationMap`] with SAMID scores.
pub fn sam_sid_classifier(
    image: &HyperspectralImage,
    references: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<ClassificationMap> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let n_classes = references.nrows();

    if n_classes == 0 {
        return Err(NdimageError::InvalidInput("references must have at least one row".into()));
    }
    if references.ncols() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "references.ncols() {} != n_bands {}",
            references.ncols(), n_bands
        )));
    }

    // Pre-compute reference norms and probability distributions.
    let ref_norms: Vec<f64> = (0..n_classes)
        .map(|c| norm(references.row(c).to_slice().unwrap_or(&[])))
        .collect();
    let ref_probs: Vec<Vec<f64>> = (0..n_classes)
        .map(|c| to_prob_dist(references.row(c).to_slice().unwrap_or(&[])))
        .collect();

    let mut labels = vec![n_classes; n_pixels];
    let mut scores = vec![f64::MAX; n_pixels];

    for p in 0..n_pixels {
        let pixel = image.data.row(p);
        let px_slice = pixel.to_slice().unwrap_or(&[]);
        let px_norm = norm(px_slice);
        let px_prob = to_prob_dist(px_slice);

        if px_norm < 1e-14 {
            continue;
        }

        for c in 0..n_classes {
            let ref_slice = references.row(c).to_slice().unwrap_or(&[]);
            // SAM component.
            let cos_theta = (dot(px_slice, ref_slice) / (px_norm * ref_norms[c].max(1e-14)))
                .clamp(-1.0, 1.0);
            let sam = cos_theta.acos();

            // SID component.
            let rp = &ref_probs[c];
            let kl_xr: f64 = (0..n_bands)
                .map(|b| {
                    let px = safe_prob(px_prob[b]);
                    let pr = safe_prob(rp[b]);
                    px * (px / pr).ln()
                })
                .sum();
            let kl_rx: f64 = (0..n_bands)
                .map(|b| {
                    let px = safe_prob(px_prob[b]);
                    let pr = safe_prob(rp[b]);
                    pr * (pr / px).ln()
                })
                .sum();
            let sid = kl_xr + kl_rx;

            let samid = sam * sid;
            if samid < scores[p] {
                scores[p] = samid;
                labels[p] = c;
            }
        }

        if scores[p] > threshold {
            labels[p] = n_classes;
        }
    }

    Ok(ClassificationMap { labels, scores, n_classes })
}

// ─────────────────────────────────────────────────────────────────────────────
// Accuracy metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute overall accuracy, per-class precision and recall from a
/// classification map and ground-truth labels.
///
/// Pixels labelled as `n_classes` (unclassified) are excluded from accuracy.
///
/// # Returns
/// `(overall_accuracy, per_class_precision [n_classes], per_class_recall [n_classes])`.
pub fn classification_accuracy(
    predicted: &ClassificationMap,
    ground_truth: &[usize],
) -> NdimageResult<(f64, Vec<f64>, Vec<f64>)> {
    let n_pixels = predicted.labels.len();
    if ground_truth.len() != n_pixels {
        return Err(NdimageError::InvalidInput(
            "ground_truth length must match number of predicted pixels".into()
        ));
    }

    let nc = predicted.n_classes;
    let mut tp = vec![0u64; nc];
    let mut fp = vec![0u64; nc];
    let mut fn_ = vec![0u64; nc];
    let mut correct = 0u64;
    let mut total = 0u64;

    for i in 0..n_pixels {
        let pred = predicted.labels[i];
        let gt = ground_truth[i];
        if pred == nc {
            continue; // Unclassified.
        }
        total += 1;
        if pred == gt {
            correct += 1;
            if pred < nc {
                tp[pred] += 1;
            }
        } else {
            if pred < nc {
                fp[pred] += 1;
            }
            if gt < nc {
                fn_[gt] += 1;
            }
        }
    }

    let oa = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

    let precision: Vec<f64> = (0..nc)
        .map(|c| {
            let denom = tp[c] + fp[c];
            if denom > 0 { tp[c] as f64 / denom as f64 } else { 0.0 }
        })
        .collect();

    let recall: Vec<f64> = (0..nc)
        .map(|c| {
            let denom = tp[c] + fn_[c];
            if denom > 0 { tp[c] as f64 / denom as f64 } else { 0.0 }
        })
        .collect();

    Ok((oa, precision, recall))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_library(n_classes: usize, n_bands: usize) -> Array2<f64> {
        let mut lib = Array2::<f64>::zeros((n_classes, n_bands));
        for c in 0..n_classes {
            for b in 0..n_bands {
                lib[[c, b]] = ((c * n_bands + b) as f64 + 1.0) / ((n_classes * n_bands) as f64);
            }
        }
        lib
    }

    fn make_image_from_library(lib: &Array2<f64>, n_reps: usize) -> HyperspectralImage {
        let (nc, nb) = (lib.nrows(), lib.ncols());
        let n_pixels = nc * n_reps;
        let mut data = Array2::<f64>::zeros((n_pixels, nb));
        for r in 0..n_reps {
            for c in 0..nc {
                for b in 0..nb {
                    data[[r * nc + c, b]] = lib[[c, b]];
                }
            }
        }
        HyperspectralImage::new(data)
    }

    #[test]
    fn test_sam_classifier_self_match() {
        let lib = make_library(3, 10);
        let img = make_image_from_library(&lib, 1);
        let result = sam_classifier(&img, &lib, f64::MAX).expect("SAM failed");
        assert_eq!(result.labels.len(), 3);
        // Each pixel should match its own class (angle ≈ 0).
        for (i, &label) in result.labels.iter().enumerate() {
            assert_eq!(label, i, "Pixel {} should map to class {}", i, i);
        }
        for &angle in &result.scores {
            assert!(angle < 0.01, "Angle {} should be near 0 for self-match", angle);
        }
    }

    #[test]
    fn test_sam_threshold_unclassified() {
        let lib = make_library(2, 8);
        let img = make_image_from_library(&lib, 2);
        let result = sam_classifier(&img, &lib, 0.0).expect("SAM threshold failed");
        // With threshold = 0, all pixels should be unclassified (angle > 0).
        for &label in &result.labels {
            assert_eq!(label, result.n_classes);
        }
    }

    #[test]
    fn test_sid_classifier_self_match() {
        let lib = make_library(3, 10);
        let img = make_image_from_library(&lib, 1);
        let result = sid_classifier(&img, &lib, f64::MAX).expect("SID failed");
        for (i, &label) in result.labels.iter().enumerate() {
            assert_eq!(label, i, "Pixel {} should match class {}", i, i);
        }
        for &sid in &result.scores {
            assert!(sid < 0.01, "SID {} should be near 0 for self-match", sid);
        }
    }

    #[test]
    fn test_scm_self_match() {
        let lib = make_library(3, 12);
        let img = make_image_from_library(&lib, 2);
        let result = spectral_correlation_mapper(&img, &lib, f64::NEG_INFINITY)
            .expect("SCM failed");
        // For first n_classes pixels (exact copies), correlation should be 1.
        for (i, (&label, &score)) in result.labels.iter().zip(result.scores.iter()).enumerate().take(3) {
            assert_eq!(label, i % 3, "Pixel {} class mismatch", i);
            assert!(score > 0.99, "SCM score {} should be near 1 for self-match", score);
        }
    }

    #[test]
    fn test_subspace_detector_full_subspace() {
        let n_bands = 5;
        // Subspace spans all of R^5 → all pixels should be detected.
        let mut sub = Array2::<f64>::zeros((n_bands, n_bands));
        for i in 0..n_bands {
            sub[[i, i]] = 1.0;
        }
        let lib = make_library(2, n_bands);
        let img = make_image_from_library(&lib, 3);
        let (dets, scores) = subspace_detector(&img, &sub, 0.99).expect("MSD failed");
        for (&d, &s) in dets.iter().zip(scores.iter()) {
            assert!(d, "All pixels should be detected in full subspace (score={})", s);
        }
    }

    #[test]
    fn test_abundance_map_to_class_winner_takes_all() {
        let mut abund = Array2::<f64>::zeros((4, 3));
        abund[[0, 0]] = 0.8;
        abund[[0, 1]] = 0.1;
        abund[[0, 2]] = 0.1;
        abund[[1, 1]] = 0.9;
        abund[[1, 0]] = 0.05;
        abund[[1, 2]] = 0.05;
        abund[[2, 2]] = 0.6;
        abund[[2, 0]] = 0.2;
        abund[[2, 1]] = 0.2;
        abund[[3, 0]] = 0.33;
        abund[[3, 1]] = 0.33;
        abund[[3, 2]] = 0.34;

        let result = abundance_map_to_class(&abund, 0.0).expect("abundance_map_to_class failed");
        assert_eq!(result.labels[0], 0);
        assert_eq!(result.labels[1], 1);
        assert_eq!(result.labels[2], 2);
        assert_eq!(result.labels[3], 2); // 0.34 wins.
    }

    #[test]
    fn test_abundance_map_unclassified_below_threshold() {
        let mut abund = Array2::<f64>::zeros((2, 2));
        abund[[0, 0]] = 0.4;
        abund[[0, 1]] = 0.6;
        abund[[1, 0]] = 0.3;
        abund[[1, 1]] = 0.2;
        let result = abundance_map_to_class(&abund, 0.5).expect("abundance_map_to_class failed");
        assert_eq!(result.labels[0], 1); // 0.6 >= 0.5.
        assert_eq!(result.labels[1], result.n_classes); // max = 0.3 < 0.5.
    }

    #[test]
    fn test_classification_accuracy() {
        let cm = ClassificationMap {
            labels: vec![0, 1, 2, 0, 1],
            scores: vec![0.1; 5],
            n_classes: 3,
        };
        let gt = vec![0, 1, 2, 1, 0];
        let (oa, prec, rec) = classification_accuracy(&cm, &gt).expect("accuracy failed");
        // 3 correct out of 5.
        assert!((oa - 0.6).abs() < 1e-10, "OA={oa}");
        assert_eq!(prec.len(), 3);
        assert_eq!(rec.len(), 3);
    }

    #[test]
    fn test_sam_sid_hybrid() {
        let lib = make_library(3, 8);
        let img = make_image_from_library(&lib, 1);
        let result = sam_sid_classifier(&img, &lib, f64::MAX).expect("SAMID failed");
        assert_eq!(result.labels.len(), 3);
    }
}
