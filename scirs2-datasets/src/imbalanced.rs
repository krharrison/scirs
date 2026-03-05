//! Imbalanced classification benchmark datasets and resampling utilities.
//!
//! This module provides:
//!
//! - [`imbalanced_classification`] – Synthetic two-class dataset with configurable
//!   class imbalance.
//! - [`synthetic_smote`]           – SMOTE-style minority-class oversampling.
//! - [`random_oversample`]         – Simple random oversampling to a target ratio.
//! - [`random_undersample`]        – Simple random undersampling to a target ratio.
//! - [`f1_score`]                  – Binary F1 score (harmonic mean of precision &
//!   recall).
//! - [`balanced_accuracy`]         – Mean per-class recall (balanced accuracy score).
//! - [`confusion_matrix`]          – Multi-class confusion matrix.
//! - [`roc_auc`]                   – Area under the ROC curve (trapezoidal rule).
//!
//! All generators are fully deterministic given a `StdRng`.

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// imbalanced_classification
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a two-class imbalanced classification dataset.
///
/// The majority class is drawn from a multivariate Gaussian centred at the
/// origin; the minority class is drawn from a Gaussian centred at
/// `(imbalance_ratio, imbalance_ratio, …)` so the two classes overlap in
/// varying degrees depending on the ratio.  Both classes share unit variance.
///
/// # Arguments
///
/// * `n_majority`       – Number of majority-class (label 0) samples.
/// * `n_minority`       – Number of minority-class (label 1) samples.
/// * `n_features`       – Feature dimensionality (must be ≥ 1).
/// * `imbalance_ratio`  – Distance between class centroids.  Larger values
///   make the classes more separable.
/// * `rng`              – Mutable RNG for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n_majority + n_minority, n_features)` and
/// `y ∈ {0, 1}`.
///
/// # Errors
///
/// Returns an error if `n_majority == 0`, `n_minority == 0`, or
/// `n_features == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::imbalanced_classification;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let (x, y) = imbalanced_classification(900, 100, 4, 2.0, &mut rng)
///     .expect("imbalanced_classification failed");
/// assert_eq!(x.shape(), &[1000, 4]);
/// assert_eq!(y.len(), 1000);
/// ```
pub fn imbalanced_classification(
    n_majority: usize,
    n_minority: usize,
    n_features: usize,
    imbalance_ratio: f64,
    rng: &mut StdRng,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_majority == 0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: n_majority must be >= 1".to_string(),
        ));
    }
    if n_minority == 0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: n_minority must be >= 1".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: n_features must be >= 1".to_string(),
        ));
    }

    let n_total = n_majority + n_minority;
    let normal = scirs2_core::random::Normal::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution failed: {e}"))
    })?;

    let mut x = Array2::zeros((n_total, n_features));
    let mut y = Array1::zeros(n_total);

    // Majority class: centroid at origin
    for i in 0..n_majority {
        for j in 0..n_features {
            x[[i, j]] = normal.sample(rng);
        }
        y[i] = 0;
    }

    // Minority class: centroid shifted by imbalance_ratio along each dimension
    for i in 0..n_minority {
        let row = n_majority + i;
        for j in 0..n_features {
            x[[row, j]] = normal.sample(rng) + imbalance_ratio;
        }
        y[row] = 1;
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// synthetic_smote
// ─────────────────────────────────────────────────────────────────────────────

/// Generate synthetic minority-class samples using a SMOTE-inspired algorithm.
///
/// For each of `n_synthetic` requested samples:
///
/// 1. Pick a random anchor sample from `minority_samples`.
/// 2. Find its `k` nearest neighbours (Euclidean distance).
/// 3. Pick one of those neighbours uniformly at random.
/// 4. Interpolate: `new = anchor + α * (neighbour − anchor)` where `α ~ U(0, 1)`.
///
/// This procedure mirrors the original SMOTE (Chawla et al., 2002).
///
/// # Arguments
///
/// * `minority_samples` – `(n, p)` array of existing minority-class samples.
///   Must have at least 2 rows.
/// * `k`                – Number of nearest neighbours to consider (must be ≥ 1
///   and < `minority_samples.nrows()`).
/// * `n_synthetic`      – Number of synthetic samples to generate.
/// * `rng`              – Mutable RNG.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_synthetic, p)`.
///
/// # Errors
///
/// Returns an error if the inputs are out of range.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::synthetic_smote;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::prelude::*;
///
/// let minority = Array2::from_shape_vec((5, 2), vec![
///     0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.2, 0.8, 0.9, 0.1,
/// ]).expect("shape");
/// let mut rng = StdRng::seed_from_u64(7);
/// let synthetic = synthetic_smote(&minority, 2, 20, &mut rng)
///     .expect("smote failed");
/// assert_eq!(synthetic.shape(), &[20, 2]);
/// ```
pub fn synthetic_smote(
    minority_samples: &Array2<f64>,
    k: usize,
    n_synthetic: usize,
    rng: &mut StdRng,
) -> Result<Array2<f64>> {
    let n = minority_samples.nrows();
    let p = minority_samples.ncols();

    if n < 2 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_smote: minority_samples must have at least 2 rows".to_string(),
        ));
    }
    if k == 0 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_smote: k must be >= 1".to_string(),
        ));
    }
    if k >= n {
        return Err(DatasetsError::InvalidFormat(format!(
            "synthetic_smote: k ({k}) must be < minority_samples.nrows() ({n})"
        )));
    }
    if n_synthetic == 0 {
        return Ok(Array2::zeros((0, p)));
    }
    if p == 0 {
        return Err(DatasetsError::InvalidFormat(
            "synthetic_smote: minority_samples must have at least 1 column".to_string(),
        ));
    }

    let alpha_dist = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform distribution failed: {e}"))
    })?;
    let anchor_dist =
        scirs2_core::random::Uniform::new(0usize, n).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform index distribution failed: {e}"))
        })?;
    let neighbour_dist =
        scirs2_core::random::Uniform::new(0usize, k).map_err(|e| {
            DatasetsError::ComputationError(format!(
                "Uniform neighbour index distribution failed: {e}"
            ))
        })?;

    let mut out = Array2::zeros((n_synthetic, p));

    for s in 0..n_synthetic {
        // Pick anchor
        let anchor_idx = anchor_dist.sample(rng);
        let anchor = minority_samples.row(anchor_idx);

        // Compute distances to all other points and find the k nearest
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&i| i != anchor_idx)
            .map(|i| {
                let row = minority_samples.row(i);
                let dist_sq: f64 = anchor
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (dist_sq, i)
            })
            .collect();

        // Partial sort: bring k smallest to the front
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Pick one of the k nearest neighbours
        let nn_pos = neighbour_dist.sample(rng);
        let nn_idx = dists[nn_pos].1;
        let neighbour = minority_samples.row(nn_idx);

        // Interpolate
        let alpha = alpha_dist.sample(rng);
        for j in 0..p {
            out[[s, j]] = anchor[j] + alpha * (neighbour[j] - anchor[j]);
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// random_oversample
// ─────────────────────────────────────────────────────────────────────────────

/// Oversample the minority class by random duplication.
///
/// The function counts the majority class (class with the most samples) and
/// the minority class.  It duplicates minority-class rows with replacement
/// until the minority-to-majority ratio reaches at least `target_ratio`.
///
/// Only binary labels (`0` and `1`) are currently supported.
///
/// # Arguments
///
/// * `x`            – Feature matrix `(n, p)`.
/// * `y`            – Label vector `(n,)`.  Must contain only `0` and `1`.
/// * `target_ratio` – Desired `n_minority / n_majority` after oversampling
///   (clamped to `[0, 1]`).  Use `1.0` for perfect balance.
/// * `rng`          – Mutable RNG.
///
/// # Returns
///
/// `(X_new, y_new)` — augmented dataset.  Original rows are preserved at the
/// start; duplicated rows are appended.
///
/// # Errors
///
/// Returns an error if `x` and `y` have inconsistent lengths, `y` contains
/// labels other than `0`/`1`, or either class is absent.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::{imbalanced_classification, random_oversample};
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = StdRng::seed_from_u64(1);
/// let (x, y) = imbalanced_classification(90, 10, 3, 1.5, &mut rng)
///     .expect("gen failed");
/// let mut rng2 = StdRng::seed_from_u64(2);
/// let (x2, y2) = random_oversample(&x, &y, 1.0, &mut rng2)
///     .expect("oversample failed");
/// // minority count should now equal majority count
/// let n_min: usize = y2.iter().filter(|&&v| v == 1).count();
/// let n_maj: usize = y2.iter().filter(|&&v| v == 0).count();
/// assert_eq!(n_min, n_maj);
/// ```
pub fn random_oversample(
    x: &Array2<f64>,
    y: &Array1<usize>,
    target_ratio: f64,
    rng: &mut StdRng,
) -> Result<(Array2<f64>, Array1<usize>)> {
    validate_xy(x, y, "random_oversample")?;
    let target_ratio = target_ratio.max(0.0).min(1.0);

    let (maj_indices, min_indices) = split_binary_classes(y, "random_oversample")?;
    let n_maj = maj_indices.len();
    let n_min = min_indices.len();

    // How many extra minority samples are needed?
    let desired_n_min = (n_maj as f64 * target_ratio).round() as usize;
    if desired_n_min <= n_min {
        // Already at or above target; return unchanged clone
        return Ok((x.to_owned(), y.to_owned()));
    }
    let n_extra = desired_n_min - n_min;

    let p = x.ncols();
    let n_out = x.nrows() + n_extra;

    let mut x_out = Array2::zeros((n_out, p));
    let mut y_out = Array1::zeros(n_out);

    // Copy original data
    for i in 0..x.nrows() {
        for j in 0..p {
            x_out[[i, j]] = x[[i, j]];
        }
        y_out[i] = y[i];
    }

    // Sample with replacement from minority indices
    let min_dist =
        scirs2_core::random::Uniform::new(0usize, n_min).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform distribution failed: {e}"))
        })?;

    for extra in 0..n_extra {
        let src_idx = min_indices[min_dist.sample(rng)];
        let row_out = x.nrows() + extra;
        for j in 0..p {
            x_out[[row_out, j]] = x[[src_idx, j]];
        }
        y_out[row_out] = 1;
    }

    Ok((x_out, y_out))
}

// ─────────────────────────────────────────────────────────────────────────────
// random_undersample
// ─────────────────────────────────────────────────────────────────────────────

/// Undersample the majority class by random removal.
///
/// The function removes majority-class rows without replacement until the
/// minority-to-majority ratio reaches at least `target_ratio`.
///
/// Only binary labels (`0` and `1`) are supported.
///
/// # Arguments
///
/// * `x`            – Feature matrix `(n, p)`.
/// * `y`            – Label vector `(n,)` with values in `{0, 1}`.
/// * `target_ratio` – Desired `n_minority / n_majority` after undersampling
///   (clamped to `[0, 1]`).  Use `1.0` for perfect balance.
/// * `rng`          – Mutable RNG.
///
/// # Returns
///
/// `(X_new, y_new)` — reduced dataset.
///
/// # Errors
///
/// Returns an error if `x` and `y` are inconsistent, labels are not binary,
/// or undersampling would eliminate all majority samples.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::{imbalanced_classification, random_undersample};
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = StdRng::seed_from_u64(1);
/// let (x, y) = imbalanced_classification(900, 100, 3, 1.5, &mut rng)
///     .expect("gen failed");
/// let mut rng2 = StdRng::seed_from_u64(3);
/// let (x2, y2) = random_undersample(&x, &y, 1.0, &mut rng2)
///     .expect("undersample failed");
/// let n_min: usize = y2.iter().filter(|&&v| v == 1).count();
/// let n_maj: usize = y2.iter().filter(|&&v| v == 0).count();
/// assert_eq!(n_min, n_maj);
/// ```
pub fn random_undersample(
    x: &Array2<f64>,
    y: &Array1<usize>,
    target_ratio: f64,
    rng: &mut StdRng,
) -> Result<(Array2<f64>, Array1<usize>)> {
    validate_xy(x, y, "random_undersample")?;
    let target_ratio = target_ratio.max(0.0).min(1.0);

    let (maj_indices, min_indices) = split_binary_classes(y, "random_undersample")?;
    let n_maj = maj_indices.len();
    let n_min = min_indices.len();

    // How many majority samples to keep?
    let desired_n_maj = if target_ratio > 0.0 {
        (n_min as f64 / target_ratio).ceil() as usize
    } else {
        0
    };
    let keep_n_maj = desired_n_maj.min(n_maj);

    if keep_n_maj == n_maj {
        return Ok((x.to_owned(), y.to_owned()));
    }

    // Shuffle majority indices and keep only the first keep_n_maj
    let mut shuffled_maj = maj_indices.clone();
    {
        use scirs2_core::random::SliceRandom;
        shuffled_maj.shuffle(rng);
    }
    shuffled_maj.truncate(keep_n_maj);

    // Build output by collecting selected rows
    let mut keep_indices: Vec<usize> = shuffled_maj;
    keep_indices.extend_from_slice(&min_indices);
    keep_indices.sort_unstable();

    let p = x.ncols();
    let n_out = keep_indices.len();
    let mut x_out = Array2::zeros((n_out, p));
    let mut y_out = Array1::zeros(n_out);

    for (out_row, &src) in keep_indices.iter().enumerate() {
        for j in 0..p {
            x_out[[out_row, j]] = x[[src, j]];
        }
        y_out[out_row] = y[src];
    }

    Ok((x_out, y_out))
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the binary F1 score for the positive class (label `1`).
///
/// `F1 = 2 * precision * recall / (precision + recall)`
///
/// Returns `0.0` when both precision and recall are zero (no true positives
/// and no positive predictions).
///
/// # Arguments
///
/// * `y_true` – Ground-truth binary labels.
/// * `y_pred` – Predicted binary labels.
///
/// # Errors
///
/// Returns an error if `y_true` and `y_pred` have different lengths.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::f1_score;
/// use scirs2_core::ndarray::array;
///
/// let y_true = array![1, 0, 1, 1, 0];
/// let y_pred = array![1, 0, 0, 1, 1];
/// let f1 = f1_score(&y_true, &y_pred).expect("f1 failed");
/// // TP=2, FP=1, FN=1 → precision=2/3, recall=2/3 → F1=2/3
/// assert!((f1 - 2.0 / 3.0).abs() < 1e-9);
/// ```
pub fn f1_score(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<f64> {
    check_same_len(y_true.len(), y_pred.len(), "f1_score")?;

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        match (t, p) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        Ok(2.0 * precision * recall / (precision + recall))
    } else {
        Ok(0.0)
    }
}

/// Compute the balanced accuracy score.
///
/// Balanced accuracy is the arithmetic mean of per-class recall (sensitivity):
///
/// ```text
/// balanced_accuracy = (recall_class_0 + recall_class_1 + …) / n_classes
/// ```
///
/// This is equivalent to `(sensitivity + specificity) / 2` in the binary case.
///
/// Classes with zero true positives contribute a recall of `0.0`.
///
/// # Arguments
///
/// * `y_true` – Ground-truth labels.
/// * `y_pred` – Predicted labels.
///
/// # Errors
///
/// Returns an error if `y_true` and `y_pred` have different lengths or are
/// empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::balanced_accuracy;
/// use scirs2_core::ndarray::array;
///
/// let y_true = array![0usize, 0, 1, 1];
/// let y_pred = array![0usize, 1, 1, 1];
/// // class-0 recall = 1/2; class-1 recall = 2/2 → balanced = 0.75
/// let ba = balanced_accuracy(&y_true, &y_pred).expect("balanced_accuracy");
/// assert!((ba - 0.75).abs() < 1e-9);
/// ```
pub fn balanced_accuracy(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<f64> {
    check_same_len(y_true.len(), y_pred.len(), "balanced_accuracy")?;
    if y_true.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "balanced_accuracy: y_true must not be empty".to_string(),
        ));
    }

    // Find unique classes
    let mut classes: Vec<usize> = y_true.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();

    let mut total_recall = 0.0_f64;
    for &c in &classes {
        let n_true_c: usize = y_true.iter().filter(|&&v| v == c).count();
        let n_correct_c: usize = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| t == c && p == c)
            .count();
        let recall_c = if n_true_c > 0 {
            n_correct_c as f64 / n_true_c as f64
        } else {
            0.0
        };
        total_recall += recall_c;
    }

    Ok(total_recall / classes.len() as f64)
}

/// Compute the multi-class confusion matrix.
///
/// The `(i, j)` entry of the returned matrix is the number of samples
/// whose true label is `i` and predicted label is `j`.
///
/// # Arguments
///
/// * `y_true`    – Ground-truth labels.
/// * `y_pred`    – Predicted labels.
/// * `n_classes` – Number of classes.  Labels must be in `0..n_classes`.
///
/// # Errors
///
/// Returns an error if `y_true` and `y_pred` have different lengths, or any
/// label value `>= n_classes`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::confusion_matrix;
/// use scirs2_core::ndarray::array;
///
/// let y_true = array![0usize, 1, 2, 0, 1];
/// let y_pred = array![0usize, 2, 2, 1, 1];
/// let cm = confusion_matrix(&y_true, &y_pred, 3).expect("confusion_matrix");
/// assert_eq!(cm[[0, 0]], 1); // true 0 predicted as 0
/// assert_eq!(cm[[1, 2]], 1); // true 1 predicted as 2
/// assert_eq!(cm[[2, 2]], 1); // true 2 predicted as 2
/// ```
pub fn confusion_matrix(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    n_classes: usize,
) -> Result<Array2<usize>> {
    check_same_len(y_true.len(), y_pred.len(), "confusion_matrix")?;
    if n_classes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "confusion_matrix: n_classes must be >= 1".to_string(),
        ));
    }

    let mut cm = Array2::zeros((n_classes, n_classes));

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t >= n_classes {
            return Err(DatasetsError::InvalidFormat(format!(
                "confusion_matrix: true label {t} >= n_classes {n_classes}"
            )));
        }
        if p >= n_classes {
            return Err(DatasetsError::InvalidFormat(format!(
                "confusion_matrix: predicted label {p} >= n_classes {n_classes}"
            )));
        }
        cm[[t, p]] += 1;
    }

    Ok(cm)
}

/// Compute the area under the Receiver Operating Characteristic (ROC) curve.
///
/// Uses the trapezoidal rule.  Scores are ranked in descending order to sweep
/// the ROC curve from the high-recall end.  Ties in the score are handled by
/// grouping them into a single trapezoid step.
///
/// # Arguments
///
/// * `y_true`  – Binary ground-truth labels (`0` = negative, `1` = positive).
/// * `y_score` – Predicted anomaly / positive-class scores.  Higher means
///   more likely positive.
///
/// # Returns
///
/// AUC in `[0, 1]`.  Returns `0.5` when there are no positive or no negative
/// examples (degenerate case).
///
/// # Errors
///
/// Returns an error if `y_true` and `y_score` have different lengths or
/// `y_true` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::imbalanced::roc_auc;
/// use scirs2_core::ndarray::array;
///
/// // Perfect classifier
/// let y_true  = array![0usize, 0, 1, 1];
/// let y_score = array![0.1, 0.2, 0.8, 0.9];
/// let auc = roc_auc(&y_true, &y_score).expect("roc_auc");
/// assert!((auc - 1.0).abs() < 1e-9, "expected 1.0, got {auc}");
/// ```
pub fn roc_auc(y_true: &Array1<usize>, y_score: &Array1<f64>) -> Result<f64> {
    check_same_len(y_true.len(), y_score.len(), "roc_auc")?;
    if y_true.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "roc_auc: arrays must not be empty".to_string(),
        ));
    }

    let n_pos: usize = y_true.iter().filter(|&&v| v == 1).count();
    let n_neg: usize = y_true.iter().filter(|&&v| v == 0).count();

    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5);
    }

    // Sort by descending score
    let mut order: Vec<usize> = (0..y_true.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sweep the ROC curve using the trapezoidal rule
    let mut auc = 0.0_f64;
    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut prev_score = f64::INFINITY;
    let mut prev_tp = 0.0_f64;
    let mut prev_fp = 0.0_f64;

    for &idx in &order {
        let score = y_score[idx];
        // When score changes, record a trapezoid
        if score != prev_score && prev_score.is_finite() {
            let tpr_cur = tp / n_pos as f64;
            let fpr_cur = fp / n_neg as f64;
            let tpr_prev = prev_tp / n_pos as f64;
            let fpr_prev = prev_fp / n_neg as f64;
            auc += (fpr_cur - fpr_prev) * (tpr_cur + tpr_prev) / 2.0;
            prev_tp = tp;
            prev_fp = fp;
        }
        prev_score = score;
        if y_true[idx] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }

    // Final segment
    let tpr_cur = tp / n_pos as f64;
    let fpr_cur = fp / n_neg as f64;
    let tpr_prev = prev_tp / n_pos as f64;
    let fpr_prev = prev_fp / n_neg as f64;
    auc += (fpr_cur - fpr_prev) * (tpr_cur + tpr_prev) / 2.0;

    Ok(auc.abs())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_xy(x: &Array2<f64>, y: &Array1<usize>, fn_name: &str) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: x.nrows() ({}) must equal y.len() ({})",
            x.nrows(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: x must not be empty"
        )));
    }
    Ok(())
}

fn check_same_len(a: usize, b: usize, fn_name: &str) -> Result<()> {
    if a != b {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: arrays must have the same length, got {a} and {b}"
        )));
    }
    Ok(())
}

/// Split binary-labelled indices into (majority, minority) groups.
///
/// "Majority" means whichever class has more samples (ties: class 0 wins).
/// Returns `(majority_indices, minority_indices)`.
fn split_binary_classes(
    y: &Array1<usize>,
    fn_name: &str,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut class0: Vec<usize> = Vec::new();
    let mut class1: Vec<usize> = Vec::new();

    for (i, &label) in y.iter().enumerate() {
        match label {
            0 => class0.push(i),
            1 => class1.push(i),
            other => {
                return Err(DatasetsError::InvalidFormat(format!(
                    "{fn_name}: label {other} is not 0 or 1"
                )))
            }
        }
    }

    if class0.is_empty() {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: class 0 is absent"
        )));
    }
    if class1.is_empty() {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: class 1 is absent"
        )));
    }

    if class0.len() >= class1.len() {
        Ok((class0, class1))
    } else {
        Ok((class1, class0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── imbalanced_classification ─────────────────────────────────────────────

    #[test]
    fn test_imbalanced_shape() {
        let mut rng = make_rng(42);
        let (x, y) =
            imbalanced_classification(900, 100, 4, 2.0, &mut rng).expect("gen failed");
        assert_eq!(x.shape(), &[1000, 4]);
        assert_eq!(y.len(), 1000);
    }

    #[test]
    fn test_imbalanced_label_counts() {
        let mut rng = make_rng(1);
        let (_, y) =
            imbalanced_classification(80, 20, 3, 1.5, &mut rng).expect("gen failed");
        let n0: usize = y.iter().filter(|&&v| v == 0).count();
        let n1: usize = y.iter().filter(|&&v| v == 1).count();
        assert_eq!(n0, 80);
        assert_eq!(n1, 20);
    }

    #[test]
    fn test_imbalanced_error_n_majority_zero() {
        let mut rng = make_rng(1);
        assert!(imbalanced_classification(0, 10, 3, 1.0, &mut rng).is_err());
    }

    #[test]
    fn test_imbalanced_error_n_minority_zero() {
        let mut rng = make_rng(1);
        assert!(imbalanced_classification(10, 0, 3, 1.0, &mut rng).is_err());
    }

    #[test]
    fn test_imbalanced_error_n_features_zero() {
        let mut rng = make_rng(1);
        assert!(imbalanced_classification(10, 5, 0, 1.0, &mut rng).is_err());
    }

    // ── synthetic_smote ───────────────────────────────────────────────────────

    #[test]
    fn test_smote_shape() {
        let minority = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.2, 0.8, 0.9, 0.1],
        )
        .expect("shape");
        let mut rng = make_rng(7);
        let syn = synthetic_smote(&minority, 2, 20, &mut rng).expect("smote");
        assert_eq!(syn.shape(), &[20, 2]);
    }

    #[test]
    fn test_smote_zero_synthetic() {
        let minority = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .expect("shape");
        let mut rng = make_rng(1);
        let syn = synthetic_smote(&minority, 2, 0, &mut rng).expect("smote zero");
        assert_eq!(syn.shape(), &[0, 2]);
    }

    #[test]
    fn test_smote_error_too_few_samples() {
        let minority = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("shape");
        let mut rng = make_rng(1);
        assert!(synthetic_smote(&minority, 1, 5, &mut rng).is_err());
    }

    #[test]
    fn test_smote_error_k_too_large() {
        let minority = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("shape");
        let mut rng = make_rng(1);
        assert!(synthetic_smote(&minority, 3, 5, &mut rng).is_err());
    }

    #[test]
    fn test_smote_interpolation_bounds() {
        // Minority samples in [0, 1]; SMOTE should stay in [0, 1] (interpolation)
        let minority = Array2::from_shape_vec(
            (4, 1),
            vec![0.0, 0.25, 0.75, 1.0],
        )
        .expect("shape");
        let mut rng = make_rng(99);
        let syn = synthetic_smote(&minority, 2, 100, &mut rng).expect("smote bounds");
        for v in syn.iter() {
            assert!(
                *v >= -1e-9 && *v <= 1.0 + 1e-9,
                "SMOTE sample {v} out of [0,1]"
            );
        }
    }

    // ── random_oversample ─────────────────────────────────────────────────────

    #[test]
    fn test_oversample_balance() {
        let mut rng = make_rng(1);
        let (x, y) =
            imbalanced_classification(90, 10, 3, 1.5, &mut rng).expect("gen");
        let mut rng2 = make_rng(2);
        let (_, y2) = random_oversample(&x, &y, 1.0, &mut rng2).expect("oversample");
        let n0: usize = y2.iter().filter(|&&v| v == 0).count();
        let n1: usize = y2.iter().filter(|&&v| v == 1).count();
        assert_eq!(n0, n1, "oversample should balance classes");
    }

    #[test]
    fn test_oversample_already_balanced() {
        let mut rng = make_rng(1);
        let (x, y) =
            imbalanced_classification(50, 50, 3, 1.0, &mut rng).expect("gen");
        let mut rng2 = make_rng(2);
        let (x2, y2) = random_oversample(&x, &y, 1.0, &mut rng2).expect("oversample noop");
        assert_eq!(x2.nrows(), x.nrows());
        assert_eq!(y2.len(), y.len());
    }

    #[test]
    fn test_oversample_error_mismatch() {
        let x = Array2::zeros((10, 3));
        let y = Array1::zeros(9usize);
        let mut rng = make_rng(1);
        assert!(random_oversample(&x, &y, 1.0, &mut rng).is_err());
    }

    // ── random_undersample ────────────────────────────────────────────────────

    #[test]
    fn test_undersample_balance() {
        let mut rng = make_rng(1);
        let (x, y) =
            imbalanced_classification(900, 100, 3, 1.5, &mut rng).expect("gen");
        let mut rng2 = make_rng(3);
        let (_, y2) = random_undersample(&x, &y, 1.0, &mut rng2).expect("undersample");
        let n0: usize = y2.iter().filter(|&&v| v == 0).count();
        let n1: usize = y2.iter().filter(|&&v| v == 1).count();
        assert_eq!(n0, n1, "undersample should balance classes");
    }

    #[test]
    fn test_undersample_preserves_minority() {
        let mut rng = make_rng(5);
        let (x, y) =
            imbalanced_classification(80, 20, 4, 1.5, &mut rng).expect("gen");
        let mut rng2 = make_rng(6);
        let (_, y2) = random_undersample(&x, &y, 1.0, &mut rng2).expect("undersample");
        let n_min: usize = y2.iter().filter(|&&v| v == 1).count();
        assert_eq!(n_min, 20, "minority class should be fully preserved");
    }

    // ── f1_score ──────────────────────────────────────────────────────────────

    #[test]
    fn test_f1_perfect() {
        let y = array![0usize, 1, 1, 0, 1];
        let f1 = f1_score(&y, &y).expect("perfect f1");
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_f1_zero_precision_recall() {
        // Predict all 0 when true is all 1
        let y_true = array![1usize, 1, 1];
        let y_pred = array![0usize, 0, 0];
        let f1 = f1_score(&y_true, &y_pred).expect("zero f1");
        assert!((f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_f1_known_value() {
        let y_true = array![1usize, 0, 1, 1, 0];
        let y_pred = array![1usize, 0, 0, 1, 1];
        let f1 = f1_score(&y_true, &y_pred).expect("known f1");
        // TP=2, FP=1, FN=1 → prec=2/3, rec=2/3 → F1=2/3
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9, "f1={f1}");
    }

    #[test]
    fn test_f1_error_length_mismatch() {
        let a = array![1usize, 0];
        let b = array![1usize, 0, 1];
        assert!(f1_score(&a, &b).is_err());
    }

    // ── balanced_accuracy ─────────────────────────────────────────────────────

    #[test]
    fn test_balanced_accuracy_perfect() {
        let y = array![0usize, 1, 0, 1];
        let ba = balanced_accuracy(&y, &y).expect("perfect ba");
        assert!((ba - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_balanced_accuracy_known() {
        let y_true = array![0usize, 0, 1, 1];
        let y_pred = array![0usize, 1, 1, 1];
        // class-0 recall=1/2, class-1 recall=2/2 → ba=0.75
        let ba = balanced_accuracy(&y_true, &y_pred).expect("known ba");
        assert!((ba - 0.75).abs() < 1e-9, "ba={ba}");
    }

    #[test]
    fn test_balanced_accuracy_error_length_mismatch() {
        let a = array![0usize, 1];
        let b = array![0usize];
        assert!(balanced_accuracy(&a, &b).is_err());
    }

    // ── confusion_matrix ──────────────────────────────────────────────────────

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 0, 1];
        let cm = confusion_matrix(&y_true, &y_pred, 2).expect("cm binary");
        assert_eq!(cm[[0, 0]], 2); // TN
        assert_eq!(cm[[1, 1]], 2); // TP
        assert_eq!(cm[[1, 0]], 1); // FN
        assert_eq!(cm[[0, 1]], 0); // FP
    }

    #[test]
    fn test_confusion_matrix_3class() {
        let y_true = array![0usize, 1, 2, 0, 1];
        let y_pred = array![0usize, 2, 2, 1, 1];
        let cm = confusion_matrix(&y_true, &y_pred, 3).expect("cm 3-class");
        assert_eq!(cm[[0, 0]], 1);
        assert_eq!(cm[[1, 2]], 1);
        assert_eq!(cm[[2, 2]], 1);
    }

    #[test]
    fn test_confusion_matrix_error_out_of_range() {
        let y_true = array![0usize, 1, 3]; // 3 >= n_classes=3
        let y_pred = array![0usize, 1, 2];
        assert!(confusion_matrix(&y_true, &y_pred, 3).is_err());
    }

    // ── roc_auc ───────────────────────────────────────────────────────────────

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.2, 0.8, 0.9];
        let auc = roc_auc(&y_true, &y_score).expect("perfect auc");
        assert!((auc - 1.0).abs() < 1e-9, "perfect auc={auc}");
    }

    #[test]
    fn test_roc_auc_random() {
        // Random classifier should produce AUC ≈ 0.5
        let y_true = array![0usize, 1, 0, 1, 0, 1];
        let y_score = array![0.5_f64, 0.5, 0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc(&y_true, &y_score).expect("random auc");
        // With all equal scores the AUC should be 0.5
        assert!((auc - 0.5).abs() < 1e-9, "random auc={auc}");
    }

    #[test]
    fn test_roc_auc_degenerate_no_positive() {
        let y_true = array![0usize, 0, 0];
        let y_score = array![0.1_f64, 0.5, 0.9];
        let auc = roc_auc(&y_true, &y_score).expect("degenerate auc");
        assert!((auc - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_roc_auc_error_empty() {
        let y_true: Array1<usize> = Array1::zeros(0);
        let y_score: Array1<f64> = Array1::zeros(0);
        assert!(roc_auc(&y_true, &y_score).is_err());
    }
}
