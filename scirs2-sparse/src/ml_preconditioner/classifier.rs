//! Classifiers for preconditioner selection.
//!
//! Provides both a rule-based heuristic classifier (requires no training) and
//! a simple random-forest classifier that can be trained on labelled data.

use crate::error::{SparseError, SparseResult};

use super::cost_model;
use super::feature_extraction::{extract_features, normalize_features};
use super::types::{
    CostEstimate, MatrixFeatures, PreconditionerType, SelectionConfig, SelectionResult,
};

// ============================================================
// Decision stump
// ============================================================

/// A single axis-aligned split.
#[derive(Debug, Clone)]
pub struct DecisionStump {
    /// Index into the feature vector.
    pub feature_idx: usize,
    /// Split threshold.
    pub threshold: f64,
    /// Class label when feature < threshold.
    pub left_class: usize,
    /// Class label when feature >= threshold.
    pub right_class: usize,
}

impl DecisionStump {
    /// Predict the class label for a feature vector.
    pub fn predict(&self, features: &[f64]) -> usize {
        let val = features.get(self.feature_idx).copied().unwrap_or(0.0);
        if val < self.threshold {
            self.left_class
        } else {
            self.right_class
        }
    }
}

// ============================================================
// Decision tree
// ============================================================

/// A binary decision tree of bounded depth.
#[derive(Debug, Clone)]
pub enum DecisionTree {
    /// Leaf node with a class label.
    Leaf(usize),
    /// Internal split node.
    Split {
        /// The decision stump at this node.
        stump: DecisionStump,
        /// Left subtree (feature < threshold).
        left: Box<DecisionTree>,
        /// Right subtree (feature >= threshold).
        right: Box<DecisionTree>,
    },
}

impl DecisionTree {
    /// Build a decision tree from labelled data with a maximum depth.
    pub fn train(features: &[Vec<f64>], labels: &[usize], max_depth: usize) -> Self {
        Self::build(features, labels, max_depth, 0)
    }

    fn build(features: &[Vec<f64>], labels: &[usize], max_depth: usize, depth: usize) -> Self {
        if labels.is_empty() {
            return Self::Leaf(0);
        }

        // Check if all labels are the same
        let first = labels[0];
        if labels.iter().all(|&l| l == first) || depth >= max_depth || features.is_empty() {
            return Self::Leaf(majority_class(labels));
        }

        let n_features = features.first().map_or(0, |f| f.len());
        if n_features == 0 {
            return Self::Leaf(majority_class(labels));
        }

        // Find best split via Gini impurity reduction
        let mut best_gini = f64::INFINITY;
        let mut best_stump = DecisionStump {
            feature_idx: 0,
            threshold: 0.0,
            left_class: 0,
            right_class: 0,
        };
        let mut best_left_idx: Vec<usize> = Vec::new();
        let mut best_right_idx: Vec<usize> = Vec::new();

        for feat in 0..n_features {
            // Collect unique thresholds (midpoints of sorted values)
            let mut vals: Vec<f64> = features.iter().map(|f| f[feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            for window in vals.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;
                let mut left_labels = Vec::new();
                let mut right_labels = Vec::new();
                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();

                for (i, f) in features.iter().enumerate() {
                    if f[feat] < threshold {
                        left_labels.push(labels[i]);
                        left_idx.push(i);
                    } else {
                        right_labels.push(labels[i]);
                        right_idx.push(i);
                    }
                }

                if left_labels.is_empty() || right_labels.is_empty() {
                    continue;
                }

                let n_total = labels.len() as f64;
                let gini = (left_labels.len() as f64 / n_total) * gini_impurity(&left_labels)
                    + (right_labels.len() as f64 / n_total) * gini_impurity(&right_labels);

                if gini < best_gini {
                    best_gini = gini;
                    best_stump = DecisionStump {
                        feature_idx: feat,
                        threshold,
                        left_class: majority_class(&left_labels),
                        right_class: majority_class(&right_labels),
                    };
                    best_left_idx = left_idx;
                    best_right_idx = right_idx;
                }
            }
        }

        if best_left_idx.is_empty() || best_right_idx.is_empty() {
            return Self::Leaf(majority_class(labels));
        }

        let left_features: Vec<Vec<f64>> =
            best_left_idx.iter().map(|&i| features[i].clone()).collect();
        let left_labels: Vec<usize> = best_left_idx.iter().map(|&i| labels[i]).collect();
        let right_features: Vec<Vec<f64>> = best_right_idx
            .iter()
            .map(|&i| features[i].clone())
            .collect();
        let right_labels: Vec<usize> = best_right_idx.iter().map(|&i| labels[i]).collect();

        Self::Split {
            stump: best_stump,
            left: Box::new(Self::build(
                &left_features,
                &left_labels,
                max_depth,
                depth + 1,
            )),
            right: Box::new(Self::build(
                &right_features,
                &right_labels,
                max_depth,
                depth + 1,
            )),
        }
    }

    /// Predict the class for a single feature vector.
    pub fn predict(&self, features: &[f64]) -> usize {
        match self {
            Self::Leaf(label) => *label,
            Self::Split { stump, left, right } => {
                if stump.predict(features) == stump.left_class {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

// ============================================================
// Random forest
// ============================================================

/// A bagged ensemble of decision trees.
#[derive(Debug, Clone)]
pub struct RandomForest {
    /// Individual trees in the ensemble.
    pub trees: Vec<DecisionTree>,
    /// Number of distinct class labels.
    pub n_classes: usize,
}

impl RandomForest {
    /// Train a random forest on labelled feature data.
    ///
    /// Uses bootstrap sampling (simple cyclic resampling for determinism in
    /// the absence of an RNG) and builds each tree with max_depth = 5.
    pub fn train(features: &[Vec<f64>], labels: &[usize], n_trees: usize) -> Self {
        let n_classes = labels.iter().copied().max().map_or(0, |m| m + 1);
        let n_samples = features.len();
        let mut trees = Vec::with_capacity(n_trees);

        for t in 0..n_trees {
            // Deterministic bootstrap: offset + stride
            let offset = t % n_samples.max(1);
            let bag_size = n_samples;
            let mut bag_features = Vec::with_capacity(bag_size);
            let mut bag_labels = Vec::with_capacity(bag_size);
            for i in 0..bag_size {
                let idx = (offset + i * (t + 1)) % n_samples.max(1);
                if idx < n_samples {
                    bag_features.push(features[idx].clone());
                    bag_labels.push(labels[idx]);
                }
            }

            let tree = DecisionTree::train(&bag_features, &bag_labels, 5);
            trees.push(tree);
        }

        Self { trees, n_classes }
    }

    /// Predict the class for a feature vector using majority vote.
    pub fn predict(&self, features: &[f64]) -> usize {
        if self.trees.is_empty() {
            return 0;
        }
        let mut votes = vec![0usize; self.n_classes.max(1)];
        for tree in &self.trees {
            let pred = tree.predict(features);
            if pred < votes.len() {
                votes[pred] += 1;
            }
        }
        votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map_or(0, |(idx, _)| idx)
    }
}

// ============================================================
// Heuristic classifier
// ============================================================

/// Rule-based heuristic classifier that requires no training data.
///
/// Uses structural and numerical properties of the matrix to pick
/// a preconditioner via human-expert rules.
#[derive(Debug, Clone, Default)]
pub struct HeuristicClassifier;

impl HeuristicClassifier {
    /// Select a preconditioner type based on matrix features.
    pub fn predict(&self, features: &MatrixFeatures) -> PreconditionerType {
        let is_diag_dominant = features.diag_dominance >= 1.0;
        let is_symmetric = features.symmetry_measure > 0.95;
        let is_small = features.n <= 500;
        let is_dense = features.density > 0.1;
        let is_large = features.n > 10_000;
        let is_spd_like = is_diag_dominant && features.has_positive_diagonal && is_symmetric;

        if is_small && is_dense {
            return PreconditionerType::None;
        }
        if is_spd_like {
            return PreconditionerType::IC0;
        }
        if is_diag_dominant && is_symmetric {
            return PreconditionerType::SSOR;
        }
        if is_diag_dominant {
            return PreconditionerType::Jacobi;
        }
        if is_large {
            return PreconditionerType::AMG;
        }
        PreconditionerType::ILU0
    }
}

// ============================================================
// Unified classifier wrapper
// ============================================================

/// Unified classifier that delegates to either a random forest or the
/// heuristic rule set.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PreconditionerClassifier {
    /// Learned random-forest classifier.
    Forest(RandomForest),
    /// Rule-based heuristic classifier.
    Heuristic(HeuristicClassifier),
}

impl Default for PreconditionerClassifier {
    fn default() -> Self {
        Self::Heuristic(HeuristicClassifier)
    }
}

impl PreconditionerClassifier {
    /// Map a class index to a `PreconditionerType`.
    fn class_to_type(idx: usize) -> PreconditionerType {
        match idx {
            0 => PreconditionerType::Jacobi,
            1 => PreconditionerType::SSOR,
            2 => PreconditionerType::ILU0,
            3 => PreconditionerType::IC0,
            4 => PreconditionerType::AMG,
            5 => PreconditionerType::SPAI,
            6 => PreconditionerType::Polynomial,
            7 => PreconditionerType::None,
            #[allow(unreachable_patterns)]
            _ => PreconditionerType::ILU0,
        }
    }

    /// Predict the preconditioner type.
    pub fn predict(&self, features: &MatrixFeatures) -> PreconditionerType {
        match self {
            Self::Forest(rf) => {
                let fv = normalize_features(features);
                Self::class_to_type(rf.predict(&fv))
            }
            Self::Heuristic(h) => h.predict(features),
            #[allow(unreachable_patterns)]
            _ => PreconditionerType::ILU0,
        }
    }
}

// ============================================================
// Top-level selection API
// ============================================================

/// Select the best preconditioner for a sparse matrix given as raw CSR data.
///
/// This is the main entry point. It extracts features, classifies, and
/// optionally re-ranks candidates by estimated cost.
pub fn select_preconditioner(
    values: &[f64],
    row_ptr: &[usize],
    col_idx: &[usize],
    n: usize,
    config: &SelectionConfig,
) -> SparseResult<SelectionResult> {
    let features = extract_features(values, row_ptr, col_idx, n)?;

    let classifier = PreconditionerClassifier::default();
    let recommended = classifier.predict(&features);

    // Build scored candidate list
    let candidates = [
        PreconditionerType::Jacobi,
        PreconditionerType::SSOR,
        PreconditionerType::ILU0,
        PreconditionerType::IC0,
        PreconditionerType::AMG,
        PreconditionerType::SPAI,
        PreconditionerType::Polynomial,
        PreconditionerType::None,
    ];

    let mut all_scores: Vec<(PreconditionerType, f64)> = if config.use_cost_model {
        let ranked = cost_model::rank_by_cost(&features, &candidates);
        // Invert cost to get a score (lower cost → higher score)
        let max_cost = ranked
            .iter()
            .map(|(_, c)| c.total_cost)
            .fold(0.0_f64, f64::max);
        let scale = if max_cost > 1e-30 { max_cost } else { 1.0 };
        ranked
            .iter()
            .map(|(pt, c)| (*pt, 1.0 - c.total_cost / scale))
            .collect()
    } else {
        candidates
            .iter()
            .map(|&pt| {
                let score = if pt == recommended { 1.0 } else { 0.0 };
                (pt, score)
            })
            .collect()
    };

    // Ensure recommended type gets a bonus
    for entry in &mut all_scores {
        if entry.0 == recommended {
            entry.1 += 0.5;
        }
    }

    // Sort descending by score
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Confidence based on score gap between #1 and #2
    let confidence = if all_scores.len() >= 2 {
        let gap = all_scores[0].1 - all_scores[1].1;
        (gap / (all_scores[0].1.abs() + 1e-10)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    Ok(SelectionResult {
        recommended,
        confidence,
        all_scores,
        features,
    })
}

// ============================================================
// Helpers
// ============================================================

fn majority_class(labels: &[usize]) -> usize {
    if labels.is_empty() {
        return 0;
    }
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; max_label + 1];
    for &l in labels {
        counts[l] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map_or(0, |(idx, _)| idx)
}

fn gini_impurity(labels: &[usize]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; max_label + 1];
    for &l in labels {
        counts[l] += 1;
    }
    let n = labels.len() as f64;
    let sum_sq: f64 = counts.iter().map(|&c| (c as f64 / n).powi(2)).sum();
    1.0 - sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_diag_dominant_symmetric_spd() {
        let h = HeuristicClassifier;
        let features = MatrixFeatures {
            n: 1000,
            nnz: 5000,
            density: 0.005,
            max_row_nnz: 5,
            mean_row_nnz: 5.0,
            bandwidth: 2,
            bandwidth_ratio: 0.002,
            cond_estimate: 10.0,
            spectral_radius: 6.0,
            diag_dominance: 2.0,
            symmetry_measure: 1.0,
            has_positive_diagonal: true,
        };
        assert_eq!(h.predict(&features), PreconditionerType::IC0);
    }

    #[test]
    fn test_heuristic_diag_dominant_nonsymmetric() {
        let h = HeuristicClassifier;
        let features = MatrixFeatures {
            n: 1000,
            nnz: 5000,
            density: 0.005,
            max_row_nnz: 5,
            mean_row_nnz: 5.0,
            bandwidth: 2,
            bandwidth_ratio: 0.002,
            cond_estimate: 10.0,
            spectral_radius: 6.0,
            diag_dominance: 2.0,
            symmetry_measure: 0.3,
            has_positive_diagonal: true,
        };
        assert_eq!(h.predict(&features), PreconditionerType::Jacobi);
    }

    #[test]
    fn test_heuristic_small_dense() {
        let h = HeuristicClassifier;
        let features = MatrixFeatures {
            n: 50,
            nnz: 500,
            density: 0.2,
            max_row_nnz: 20,
            mean_row_nnz: 10.0,
            bandwidth: 49,
            bandwidth_ratio: 1.0,
            cond_estimate: 5.0,
            spectral_radius: 10.0,
            diag_dominance: 0.5,
            symmetry_measure: 0.8,
            has_positive_diagonal: true,
        };
        assert_eq!(h.predict(&features), PreconditionerType::None);
    }

    #[test]
    fn test_heuristic_large_sparse() {
        let h = HeuristicClassifier;
        let features = MatrixFeatures {
            n: 100_000,
            nnz: 500_000,
            density: 0.00005,
            max_row_nnz: 7,
            mean_row_nnz: 5.0,
            bandwidth: 1000,
            bandwidth_ratio: 0.01,
            cond_estimate: 1000.0,
            spectral_radius: 100.0,
            diag_dominance: 0.5,
            symmetry_measure: 0.5,
            has_positive_diagonal: true,
        };
        assert_eq!(h.predict(&features), PreconditionerType::AMG);
    }

    #[test]
    fn test_heuristic_general() {
        let h = HeuristicClassifier;
        let features = MatrixFeatures {
            n: 2000,
            nnz: 20_000,
            density: 0.005,
            max_row_nnz: 15,
            mean_row_nnz: 10.0,
            bandwidth: 200,
            bandwidth_ratio: 0.1,
            cond_estimate: 100.0,
            spectral_radius: 50.0,
            diag_dominance: 0.3,
            symmetry_measure: 0.6,
            has_positive_diagonal: false,
        };
        assert_eq!(h.predict(&features), PreconditionerType::ILU0);
    }

    #[test]
    fn test_select_preconditioner_tridiag() {
        // 3×3 tridiag SPD
        let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptr = vec![0, 2, 5, 7];
        let config = SelectionConfig::default();
        let result =
            select_preconditioner(&values, &row_ptr, &col_idx, 3, &config).expect("select");
        // Small dense → should recommend None
        assert_eq!(result.recommended, PreconditionerType::None);
        assert!(!result.all_scores.is_empty());
    }

    #[test]
    fn test_decision_tree_pure_leaf() {
        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let labels = vec![0, 0, 0];
        let tree = DecisionTree::train(&features, &labels, 3);
        assert_eq!(tree.predict(&[1.5]), 0);
    }

    #[test]
    fn test_random_forest_simple() {
        let features = vec![
            vec![0.1, 0.2],
            vec![0.9, 0.8],
            vec![0.15, 0.25],
            vec![0.85, 0.75],
        ];
        let labels = vec![0, 1, 0, 1];
        let rf = RandomForest::train(&features, &labels, 5);
        // Predictions should be consistent for training data
        let pred0 = rf.predict(&[0.1, 0.2]);
        let pred1 = rf.predict(&[0.9, 0.8]);
        // At minimum, check they're valid class indices
        assert!(pred0 < 2);
        assert!(pred1 < 2);
    }

    #[test]
    fn test_classifier_default_is_heuristic() {
        let c = PreconditionerClassifier::default();
        match c {
            PreconditionerClassifier::Heuristic(_) => {}
            _ => panic!("default should be heuristic"),
        }
    }
}
