//! Text classification functionality
//!
//! This module provides comprehensive tools for text classification:
//!
//! - **Naive Bayes**: Multinomial and Bernoulli variants
//! - **TF-IDF + Cosine Similarity**: k-NN style classification
//! - **Feature Hashing**: Memory-efficient hashing trick for large vocabularies
//! - **Multi-label**: Support for texts belonging to multiple categories
//! - **Cross-validation**: k-fold evaluation utilities
//! - **Metrics**: Precision, recall, F1, accuracy
//! - **Feature Selection**: Document frequency based filtering
//! - **Pipelines**: End-to-end TF-IDF + classify pipelines

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::SeedableRng;
use std::collections::{HashMap, HashSet};

// ─── Feature Selector ────────────────────────────────────────────────────────

/// Text feature selector
///
/// Filters features based on document frequency.
#[derive(Debug, Clone)]
pub struct TextFeatureSelector {
    /// Minimum document frequency (fraction or count)
    min_df: f64,
    /// Maximum document frequency (fraction or count)
    max_df: f64,
    /// Whether to use raw counts instead of fractions
    use_counts: bool,
    /// Selected feature indices
    selected_features: Option<Vec<usize>>,
}

impl Default for TextFeatureSelector {
    fn default() -> Self {
        Self {
            min_df: 0.0,
            max_df: 1.0,
            use_counts: false,
            selected_features: None,
        }
    }
}

impl TextFeatureSelector {
    /// Create a new feature selector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum document frequency
    pub fn set_min_df(mut self, mindf: f64) -> Result<Self> {
        if mindf < 0.0 {
            return Err(TextError::InvalidInput(
                "min_df must be non-negative".to_string(),
            ));
        }
        self.min_df = mindf;
        Ok(self)
    }

    /// Set maximum document frequency
    pub fn set_max_df(mut self, maxdf: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&maxdf) {
            return Err(TextError::InvalidInput(
                "max_df must be between 0 and 1 for fractions".to_string(),
            ));
        }
        self.max_df = maxdf;
        Ok(self)
    }

    /// Set maximum document frequency (alias for set_max_df)
    pub fn set_max_features(self, maxfeatures: f64) -> Result<Self> {
        self.set_max_df(maxfeatures)
    }

    /// Set to use absolute counts instead of fractions
    pub fn use_counts(mut self, usecounts: bool) -> Self {
        self.use_counts = usecounts;
        self
    }

    /// Fit the feature selector to data
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut document_frequencies = vec![0; n_features];

        for sample in x.axis_iter(Axis(0)) {
            for (feature_idx, &value) in sample.iter().enumerate() {
                if value > 0.0 {
                    document_frequencies[feature_idx] += 1;
                }
            }
        }

        let min_count = if self.use_counts {
            self.min_df
        } else {
            self.min_df * n_samples as f64
        };

        let max_count = if self.use_counts {
            self.max_df
        } else {
            self.max_df * n_samples as f64
        };

        let mut selected_features = Vec::new();
        for (idx, &df) in document_frequencies.iter().enumerate() {
            let df_f64 = df as f64;
            if df_f64 >= min_count && df_f64 <= max_count {
                selected_features.push(idx);
            }
        }

        self.selected_features = Some(selected_features);
        Ok(self)
    }

    /// Transform data using selected features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected_features = self
            .selected_features
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("Feature selector not fitted".to_string()))?;

        if selected_features.is_empty() {
            return Err(TextError::InvalidInput(
                "No features selected. Try adjusting min_df and max_df".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_selected = selected_features.len();

        let mut result = Array2::zeros((n_samples, n_selected));

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            for (j, &feature_idx) in selected_features.iter().enumerate() {
                result[[i, j]] = row[feature_idx];
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features.as_ref()
    }
}

// ─── Classification Metrics ──────────────────────────────────────────────────

/// Text classification metrics
#[derive(Debug, Clone)]
pub struct TextClassificationMetrics;

impl Default for TextClassificationMetrics {
    fn default() -> Self {
        Self
    }
}

impl TextClassificationMetrics {
    /// Create a new metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate precision score
    pub fn precision<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let positive_class = class_idx.unwrap_or_default();

        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        let mut true_positives = 0;
        let mut predicted_positives = 0;

        for i in 0..predictions.len() {
            if predictions[i] == positive_class {
                predicted_positives += 1;
                if true_labels[i] == positive_class {
                    true_positives += 1;
                }
            }
        }

        if predicted_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / predicted_positives as f64)
    }

    /// Calculate recall score
    pub fn recall<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let positive_class = class_idx.unwrap_or_default();

        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        let mut true_positives = 0;
        let mut actual_positives = 0;

        for i in 0..predictions.len() {
            if true_labels[i] == positive_class {
                actual_positives += 1;
                if predictions[i] == positive_class {
                    true_positives += 1;
                }
            }
        }

        if actual_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / actual_positives as f64)
    }

    /// Calculate F1 score
    pub fn f1_score<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let precision = self.precision(predictions, true_labels, class_idx)?;
        let recall = self.recall(predictions, true_labels, class_idx)?;

        if precision + recall == 0.0 {
            return Ok(0.0);
        }

        Ok(2.0 * precision * recall / (precision + recall))
    }

    /// Calculate accuracy from predictions and true labels
    pub fn accuracy<T>(&self, predictions: &[T], truelabels: &[T]) -> Result<f64>
    where
        T: PartialEq,
    {
        if predictions.len() != truelabels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot calculate accuracy for empty arrays".to_string(),
            ));
        }

        let correct = predictions
            .iter()
            .zip(truelabels.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();

        Ok(correct as f64 / predictions.len() as f64)
    }

    /// Calculate precision, recall, and F1 score for binary classification
    pub fn binary_metrics<T>(&self, predictions: &[T], truelabels: &[T]) -> Result<(f64, f64, f64)>
    where
        T: PartialEq + Copy + Default + PartialEq<usize>,
    {
        if predictions.len() != truelabels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        for (pred, true_label) in predictions.iter().zip(truelabels.iter()) {
            if *pred == 1 && *true_label == 1 {
                tp += 1;
            } else if *pred == 1 && *true_label == 0 {
                fp += 1;
            } else if *pred == 0 && *true_label == 1 {
                fn_ += 1;
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

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok((precision, recall, f1))
    }
}

// ─── Text Dataset ────────────────────────────────────────────────────────────

/// Text classification dataset
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// The text samples
    pub texts: Vec<String>,
    /// The labels for each text
    pub labels: Vec<String>,
    /// Index mapping for labels
    label_index: Option<HashMap<String, usize>>,
}

impl TextDataset {
    /// Create a new text dataset
    pub fn new(texts: Vec<String>, labels: Vec<String>) -> Result<Self> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(
                "Texts and labels must have the same length".to_string(),
            ));
        }

        Ok(Self {
            texts,
            labels,
            label_index: None,
        })
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Get the unique labels in the dataset
    pub fn unique_labels(&self) -> Vec<String> {
        let mut unique = HashSet::new();
        for label in &self.labels {
            unique.insert(label.clone());
        }
        unique.into_iter().collect()
    }

    /// Build a label index mapping
    pub fn build_label_index(&mut self) -> Result<&mut Self> {
        let mut index = HashMap::new();
        let unique_labels = self.unique_labels();

        for (i, label) in unique_labels.iter().enumerate() {
            index.insert(label.clone(), i);
        }

        self.label_index = Some(index);
        Ok(self)
    }

    /// Get label indices
    pub fn get_label_indices(&self) -> Result<Vec<usize>> {
        let index = self
            .label_index
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("Label index not built".to_string()))?;

        self.labels
            .iter()
            .map(|label| {
                index
                    .get(label)
                    .copied()
                    .ok_or_else(|| TextError::InvalidInput(format!("Unknown label: {label}")))
            })
            .collect()
    }

    /// Split the dataset into train and test sets
    pub fn train_test_split(
        &self,
        test_size: f64,
        random_seed: Option<u64>,
    ) -> Result<(Self, Self)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(TextError::InvalidInput(
                "test_size must be between 0 and 1".to_string(),
            ));
        }

        if self.is_empty() {
            return Err(TextError::InvalidInput("Dataset is empty".to_string()));
        }

        let mut indices: Vec<usize> = (0..self.len()).collect();

        if let Some(seed) = random_seed {
            let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        } else {
            let mut rng = scirs2_core::random::rng();
            indices.shuffle(&mut rng);
        }

        let test_count = (self.len() as f64 * test_size).ceil() as usize;
        let test_indices = indices[0..test_count].to_vec();
        let train_indices = indices[test_count..].to_vec();

        let train_texts = train_indices
            .iter()
            .map(|&i| self.texts[i].clone())
            .collect();
        let train_labels = train_indices
            .iter()
            .map(|&i| self.labels[i].clone())
            .collect();
        let test_texts = test_indices
            .iter()
            .map(|&i| self.texts[i].clone())
            .collect();
        let test_labels = test_indices
            .iter()
            .map(|&i| self.labels[i].clone())
            .collect();

        let mut train_dataset = Self::new(train_texts, train_labels)?;
        let mut test_dataset = Self::new(test_texts, test_labels)?;

        if self.label_index.is_some() {
            train_dataset.build_label_index()?;
            test_dataset.build_label_index()?;
        }

        Ok((train_dataset, test_dataset))
    }
}

// ─── Classification Pipeline ─────────────────────────────────────────────────

/// Pipeline for text classification
pub struct TextClassificationPipeline {
    /// The vectorizer to use
    vectorizer: TfidfVectorizer,
    /// Optional feature selector
    feature_selector: Option<TextFeatureSelector>,
}

impl TextClassificationPipeline {
    /// Create a new pipeline with a default TF-IDF vectorizer
    pub fn with_tfidf() -> Self {
        Self::new(TfidfVectorizer::default())
    }

    /// Create a new pipeline with the given vectorizer
    pub fn new(vectorizer: TfidfVectorizer) -> Self {
        Self {
            vectorizer,
            feature_selector: None,
        }
    }

    /// Add a feature selector to the pipeline
    pub fn with_feature_selector(mut self, selector: TextFeatureSelector) -> Self {
        self.feature_selector = Some(selector);
        self
    }

    /// Fit the pipeline to training data
    pub fn fit(&mut self, dataset: &TextDataset) -> Result<&mut Self> {
        let texts: Vec<&str> = dataset.texts.iter().map(AsRef::as_ref).collect();
        self.vectorizer.fit(&texts)?;
        Ok(self)
    }

    /// Transform text data using the pipeline
    pub fn transform(&self, dataset: &TextDataset) -> Result<Array2<f64>> {
        let texts: Vec<&str> = dataset.texts.iter().map(AsRef::as_ref).collect();
        let mut features = self.vectorizer.transform_batch(&texts)?;

        if let Some(selector) = &self.feature_selector {
            features = selector.transform(&features)?;
        }

        Ok(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, dataset: &TextDataset) -> Result<Array2<f64>> {
        self.fit(dataset)?;
        self.transform(dataset)
    }
}

// ─── Multinomial Naive Bayes Classifier ──────────────────────────────────────

/// Multinomial Naive Bayes classifier for text
///
/// Suitable for text classification with word count / TF-IDF features.
/// Implements Laplace smoothing.
#[derive(Debug, Clone)]
pub struct MultinomialNaiveBayes {
    /// Word log-probabilities per class: class -> feature_idx -> log(P(w|c))
    feature_log_probs: HashMap<String, Vec<f64>>,
    /// Prior log-probabilities: class -> log(P(c))
    class_log_priors: HashMap<String, f64>,
    /// Number of features
    n_features: usize,
    /// Laplace smoothing parameter
    alpha: f64,
    /// Classes
    classes: Vec<String>,
}

impl MultinomialNaiveBayes {
    /// Create a new multinomial Naive Bayes classifier
    pub fn new(alpha: f64) -> Self {
        Self {
            feature_log_probs: HashMap::new(),
            class_log_priors: HashMap::new(),
            n_features: 0,
            alpha,
            classes: Vec::new(),
        }
    }

    /// Train the classifier
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples x n_features), e.g. TF-IDF
    /// * `labels` - Class labels for each sample
    pub fn fit(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<()> {
        if features.nrows() != labels.len() {
            return Err(TextError::InvalidInput(
                "Features and labels must have the same number of rows".into(),
            ));
        }

        let n_samples = features.nrows();
        self.n_features = features.ncols();

        // Determine classes
        let mut class_set = HashSet::new();
        for label in labels {
            class_set.insert(label.clone());
        }
        self.classes = class_set.into_iter().collect();
        self.classes.sort();

        // Compute per-class statistics
        for class in &self.classes {
            // Gather rows belonging to this class
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, l)| *l == class)
                .map(|(i, _)| i)
                .collect();

            let class_count = class_indices.len();

            // Prior log-probability
            let log_prior = (class_count as f64 / n_samples as f64).ln();
            self.class_log_priors.insert(class.clone(), log_prior);

            // Sum features for this class
            let mut feature_sums = vec![0.0; self.n_features];
            for &idx in &class_indices {
                for j in 0..self.n_features {
                    feature_sums[j] += features[[idx, j]];
                }
            }

            // Total count for this class (with smoothing)
            let total: f64 = feature_sums.iter().sum::<f64>() + self.alpha * self.n_features as f64;

            // Log probabilities with Laplace smoothing
            let log_probs: Vec<f64> = feature_sums
                .iter()
                .map(|&count| ((count + self.alpha) / total).ln())
                .collect();

            self.feature_log_probs.insert(class.clone(), log_probs);
        }

        Ok(())
    }

    /// Predict class labels for feature matrix
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<String>> {
        let mut predictions = Vec::with_capacity(features.nrows());

        for row in features.axis_iter(Axis(0)) {
            let (label, _) = self.predict_single(&row.to_owned())?;
            predictions.push(label);
        }

        Ok(predictions)
    }

    /// Predict a single sample
    fn predict_single(&self, features: &Array1<f64>) -> Result<(String, f64)> {
        if self.classes.is_empty() {
            return Err(TextError::ModelNotFitted("Classifier not trained".into()));
        }

        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes {
            let log_prior = self
                .class_log_priors
                .get(class)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);

            let log_probs = self
                .feature_log_probs
                .get(class)
                .ok_or_else(|| TextError::RuntimeError("Missing feature probs".into()))?;

            let log_likelihood: f64 = features
                .iter()
                .zip(log_probs.iter())
                .map(|(&feat, &log_p)| feat * log_p)
                .sum();

            let score = log_prior + log_likelihood;
            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        Ok((best_class, best_score))
    }
}

// ─── Bernoulli Naive Bayes Classifier ────────────────────────────────────────

/// Bernoulli Naive Bayes classifier
///
/// Works with binary features (word present/absent).
/// Suitable for short texts or bag-of-words with binary encoding.
#[derive(Debug, Clone)]
pub struct BernoulliNaiveBayes {
    /// Log probability that feature is present for each class
    feature_log_probs: HashMap<String, Vec<f64>>,
    /// Log probability that feature is absent for each class
    feature_log_neg_probs: HashMap<String, Vec<f64>>,
    /// Prior log-probabilities
    class_log_priors: HashMap<String, f64>,
    /// Number of features
    n_features: usize,
    /// Smoothing parameter
    alpha: f64,
    /// Binarization threshold
    binarize_threshold: f64,
    /// Classes
    classes: Vec<String>,
}

impl BernoulliNaiveBayes {
    /// Create a new Bernoulli Naive Bayes classifier
    pub fn new(alpha: f64) -> Self {
        Self {
            feature_log_probs: HashMap::new(),
            feature_log_neg_probs: HashMap::new(),
            class_log_priors: HashMap::new(),
            n_features: 0,
            alpha,
            binarize_threshold: 0.0,
            classes: Vec::new(),
        }
    }

    /// Set the binarization threshold
    pub fn with_binarize_threshold(mut self, threshold: f64) -> Self {
        self.binarize_threshold = threshold;
        self
    }

    /// Train the classifier
    pub fn fit(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<()> {
        if features.nrows() != labels.len() {
            return Err(TextError::InvalidInput(
                "Features and labels must have the same number of rows".into(),
            ));
        }

        let n_samples = features.nrows();
        self.n_features = features.ncols();

        let mut class_set = HashSet::new();
        for label in labels {
            class_set.insert(label.clone());
        }
        self.classes = class_set.into_iter().collect();
        self.classes.sort();

        for class in &self.classes {
            let class_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, l)| *l == class)
                .map(|(i, _)| i)
                .collect();

            let class_count = class_indices.len() as f64;

            let log_prior = (class_count / n_samples as f64).ln();
            self.class_log_priors.insert(class.clone(), log_prior);

            // Count documents where each feature is present
            let mut feature_present = vec![0.0; self.n_features];
            for &idx in &class_indices {
                for j in 0..self.n_features {
                    if features[[idx, j]] > self.binarize_threshold {
                        feature_present[j] += 1.0;
                    }
                }
            }

            // P(feature_j = 1 | class) with smoothing
            let log_probs: Vec<f64> = feature_present
                .iter()
                .map(|&count| ((count + self.alpha) / (class_count + 2.0 * self.alpha)).ln())
                .collect();

            let log_neg_probs: Vec<f64> = feature_present
                .iter()
                .map(|&count| {
                    ((class_count - count + self.alpha) / (class_count + 2.0 * self.alpha)).ln()
                })
                .collect();

            self.feature_log_probs.insert(class.clone(), log_probs);
            self.feature_log_neg_probs
                .insert(class.clone(), log_neg_probs);
        }

        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<String>> {
        let mut predictions = Vec::with_capacity(features.nrows());

        for row in features.axis_iter(Axis(0)) {
            let label = self.predict_single(&row.to_owned())?;
            predictions.push(label);
        }

        Ok(predictions)
    }

    fn predict_single(&self, features: &Array1<f64>) -> Result<String> {
        if self.classes.is_empty() {
            return Err(TextError::ModelNotFitted("Classifier not trained".into()));
        }

        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for class in &self.classes {
            let log_prior = self
                .class_log_priors
                .get(class)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);

            let log_probs = self
                .feature_log_probs
                .get(class)
                .ok_or_else(|| TextError::RuntimeError("Missing probs".into()))?;
            let log_neg_probs = self
                .feature_log_neg_probs
                .get(class)
                .ok_or_else(|| TextError::RuntimeError("Missing neg probs".into()))?;

            let mut log_likelihood = 0.0;
            for j in 0..self.n_features {
                if features[j] > self.binarize_threshold {
                    log_likelihood += log_probs[j];
                } else {
                    log_likelihood += log_neg_probs[j];
                }
            }

            let score = log_prior + log_likelihood;
            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        Ok(best_class)
    }
}

// ─── TF-IDF Cosine Similarity Classifier ─────────────────────────────────────

/// TF-IDF + cosine similarity k-NN classifier
///
/// Classifies text by finding the k nearest training examples
/// (by cosine similarity) and taking a majority vote.
pub struct TfidfCosineClassifier {
    /// Training TF-IDF vectors
    train_vectors: Option<Array2<f64>>,
    /// Training labels
    train_labels: Vec<String>,
    /// Number of neighbors
    k: usize,
}

impl TfidfCosineClassifier {
    /// Create a new TF-IDF cosine similarity classifier
    pub fn new(k: usize) -> Self {
        Self {
            train_vectors: None,
            train_labels: Vec::new(),
            k,
        }
    }

    /// Fit the classifier with pre-computed TF-IDF features
    pub fn fit(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<()> {
        if features.nrows() != labels.len() {
            return Err(TextError::InvalidInput(
                "Features and labels must have the same number of rows".into(),
            ));
        }

        self.train_vectors = Some(features.clone());
        self.train_labels = labels.to_vec();
        Ok(())
    }

    /// Predict class labels for test features
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<String>> {
        let train_vectors = self
            .train_vectors
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("Classifier not trained".into()))?;

        let mut predictions = Vec::with_capacity(features.nrows());

        for row in features.axis_iter(Axis(0)) {
            let query = row.to_owned();

            // Compute cosine similarity with all training samples
            let mut similarities: Vec<(usize, f64)> = Vec::with_capacity(train_vectors.nrows());

            for (idx, train_row) in train_vectors.axis_iter(Axis(0)).enumerate() {
                let sim = cosine_similarity(&query, &train_row.to_owned());
                similarities.push((idx, sim));
            }

            // Sort by similarity descending
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k and do majority vote
            let mut class_votes: HashMap<&str, usize> = HashMap::new();
            let k = self.k.min(similarities.len());

            for &(idx, _) in similarities.iter().take(k) {
                *class_votes.entry(&self.train_labels[idx]).or_insert(0) += 1;
            }

            let best_class = class_votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(label, _)| label.to_string())
                .unwrap_or_default();

            predictions.push(best_class);
        }

        Ok(predictions)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ─── Feature Hashing ─────────────────────────────────────────────────────────

/// Feature hasher (hashing trick) for text classification
///
/// Maps tokens to a fixed-size feature vector using hashing,
/// avoiding the need to maintain a vocabulary dictionary.
/// This is memory-efficient for large vocabularies.
pub struct FeatureHasher {
    /// Number of output features (hash buckets)
    n_features: usize,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl std::fmt::Debug for FeatureHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureHasher")
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl FeatureHasher {
    /// Create a new feature hasher with the specified number of features
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Hash a string to a feature index using FNV-1a
    fn hash_feature(&self, token: &str) -> usize {
        let mut hash: u64 = 2166136261;
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        (hash % (self.n_features as u64)) as usize
    }

    /// Determine sign from hash (for signed hashing to reduce collision bias)
    fn hash_sign(&self, token: &str) -> f64 {
        let mut hash: u64 = 84696351;
        for byte in token.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        if hash.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    /// Transform a single text into a hashed feature vector
    pub fn transform_text(&self, text: &str) -> Result<Array1<f64>> {
        let tokens = self.tokenizer.tokenize(text)?;
        let mut features = Array1::zeros(self.n_features);

        for token in &tokens {
            let idx = self.hash_feature(&token.to_lowercase());
            let sign = self.hash_sign(&token.to_lowercase());
            features[idx] += sign;
        }

        Ok(features)
    }

    /// Transform multiple texts into a feature matrix
    pub fn transform_batch(&self, texts: &[&str]) -> Result<Array2<f64>> {
        let mut matrix = Array2::zeros((texts.len(), self.n_features));

        for (i, &text) in texts.iter().enumerate() {
            let features = self.transform_text(text)?;
            for j in 0..self.n_features {
                matrix[[i, j]] = features[j];
            }
        }

        Ok(matrix)
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.n_features
    }
}

// ─── Multi-Label Classification ──────────────────────────────────────────────

/// Multi-label prediction result
#[derive(Debug, Clone)]
pub struct MultiLabelPrediction {
    /// The predicted labels (can be multiple)
    pub labels: Vec<String>,
    /// Confidence scores for each label
    pub scores: HashMap<String, f64>,
}

/// Multi-label classifier using binary relevance approach
///
/// Trains one binary classifier per label, allowing texts
/// to belong to multiple categories.
#[derive(Debug, Clone)]
pub struct MultiLabelClassifier {
    /// One binary Naive Bayes per label
    classifiers: HashMap<String, MultinomialNaiveBayes>,
    /// Prediction threshold
    threshold: f64,
    /// All known labels
    all_labels: Vec<String>,
}

impl MultiLabelClassifier {
    /// Create a new multi-label classifier
    pub fn new(threshold: f64) -> Self {
        Self {
            classifiers: HashMap::new(),
            threshold,
            all_labels: Vec::new(),
        }
    }

    /// Train the classifier
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples x n_features)
    /// * `label_sets` - For each sample, a set of labels it belongs to
    pub fn fit(&mut self, features: &Array2<f64>, label_sets: &[Vec<String>]) -> Result<()> {
        if features.nrows() != label_sets.len() {
            return Err(TextError::InvalidInput(
                "Features and label_sets must have the same length".into(),
            ));
        }

        // Collect all unique labels
        let mut all_labels_set = HashSet::new();
        for labels in label_sets {
            for label in labels {
                all_labels_set.insert(label.clone());
            }
        }
        self.all_labels = all_labels_set.into_iter().collect();
        self.all_labels.sort();

        // Train one binary classifier per label
        for label in &self.all_labels {
            let binary_labels: Vec<String> = label_sets
                .iter()
                .map(|ls| {
                    if ls.contains(label) {
                        "positive".to_string()
                    } else {
                        "negative".to_string()
                    }
                })
                .collect();

            let mut clf = MultinomialNaiveBayes::new(1.0);
            clf.fit(features, &binary_labels)?;
            self.classifiers.insert(label.clone(), clf);
        }

        Ok(())
    }

    /// Predict labels for feature matrix
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<MultiLabelPrediction>> {
        let mut predictions = Vec::with_capacity(features.nrows());

        for row in features.axis_iter(Axis(0)) {
            let row_arr = row.to_owned();
            let mut labels = Vec::new();
            let mut scores = HashMap::new();

            // Create a 1-row matrix for the classifier
            let single_row = Array2::from_shape_fn((1, row_arr.len()), |(_, j)| row_arr[j]);

            for label in &self.all_labels {
                if let Some(clf) = self.classifiers.get(label) {
                    let pred = clf.predict(&single_row)?;
                    if !pred.is_empty() && pred[0] == "positive" {
                        labels.push(label.clone());
                        scores.insert(label.clone(), 1.0);
                    } else {
                        scores.insert(label.clone(), 0.0);
                    }
                }
            }

            predictions.push(MultiLabelPrediction { labels, scores });
        }

        Ok(predictions)
    }
}

// ─── Cross-Validation ────────────────────────────────────────────────────────

/// Result of a cross-validation fold
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Fold index
    pub fold: usize,
    /// Accuracy on the fold
    pub accuracy: f64,
    /// Predictions for this fold
    pub predictions: Vec<String>,
    /// True labels for this fold
    pub true_labels: Vec<String>,
}

/// Result of cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Results for each fold
    pub fold_results: Vec<FoldResult>,
    /// Mean accuracy across folds
    pub mean_accuracy: f64,
    /// Standard deviation of accuracy
    pub std_accuracy: f64,
}

/// Perform k-fold cross-validation with multinomial Naive Bayes
///
/// # Arguments
/// * `features` - Feature matrix
/// * `labels` - Labels
/// * `k` - Number of folds
/// * `alpha` - Naive Bayes smoothing parameter
/// * `seed` - Optional random seed for reproducibility
pub fn cross_validate_nb(
    features: &Array2<f64>,
    labels: &[String],
    k: usize,
    alpha: f64,
    seed: Option<u64>,
) -> Result<CrossValidationResult> {
    if features.nrows() != labels.len() {
        return Err(TextError::InvalidInput(
            "Features and labels must have the same length".into(),
        ));
    }

    let n = features.nrows();
    if k < 2 || k > n {
        return Err(TextError::InvalidInput(format!(
            "k must be between 2 and {} (number of samples)",
            n
        )));
    }

    // Create shuffled indices
    let mut indices: Vec<usize> = (0..n).collect();
    if let Some(s) = seed {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(s);
        indices.shuffle(&mut rng);
    } else {
        let mut rng = scirs2_core::random::rng();
        indices.shuffle(&mut rng);
    }

    let fold_size = n / k;
    let mut fold_results = Vec::with_capacity(k);

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 {
            n
        } else {
            (fold + 1) * fold_size
        };

        let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, &idx)| idx)
            .collect();

        // Build train/test sets
        let n_train = train_indices.len();
        let n_test = test_indices.len();
        let n_features = features.ncols();

        let mut train_features = Array2::zeros((n_train, n_features));
        let mut train_labels = Vec::with_capacity(n_train);

        for (i, &idx) in train_indices.iter().enumerate() {
            for j in 0..n_features {
                train_features[[i, j]] = features[[idx, j]];
            }
            train_labels.push(labels[idx].clone());
        }

        let mut test_features = Array2::zeros((n_test, n_features));
        let mut test_labels = Vec::with_capacity(n_test);

        for (i, &idx) in test_indices.iter().enumerate() {
            for j in 0..n_features {
                test_features[[i, j]] = features[[idx, j]];
            }
            test_labels.push(labels[idx].clone());
        }

        // Train and predict
        let mut clf = MultinomialNaiveBayes::new(alpha);
        clf.fit(&train_features, &train_labels)?;
        let predictions = clf.predict(&test_features)?;

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(test_labels.iter())
            .filter(|(p, t)| p == t)
            .count();
        let accuracy = correct as f64 / n_test as f64;

        fold_results.push(FoldResult {
            fold,
            accuracy,
            predictions,
            true_labels: test_labels,
        });
    }

    // Compute mean and std
    let accuracies: Vec<f64> = fold_results.iter().map(|f| f.accuracy).collect();
    let mean_accuracy = accuracies.iter().sum::<f64>() / k as f64;
    let variance = accuracies
        .iter()
        .map(|&a| (a - mean_accuracy).powi(2))
        .sum::<f64>()
        / k as f64;
    let std_accuracy = variance.sqrt();

    Ok(CrossValidationResult {
        fold_results,
        mean_accuracy,
        std_accuracy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_dataset() {
        let texts = vec![
            "This is document 1".to_string(),
            "Another document".to_string(),
            "A third document".to_string(),
        ];
        let labels = vec!["A".to_string(), "B".to_string(), "A".to_string()];

        let mut dataset = TextDataset::new(texts, labels).expect("Operation failed");

        let mut label_index = HashMap::new();
        label_index.insert("A".to_string(), 0);
        label_index.insert("B".to_string(), 1);
        dataset.label_index = Some(label_index);

        let label_indices = dataset.get_label_indices().expect("Operation failed");
        assert_eq!(label_indices[0], 0);
        assert_eq!(label_indices[1], 1);
        assert_eq!(label_indices[2], 0);

        let unique_labels = dataset.unique_labels();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_train_test_split() {
        let texts = (0..10).map(|i| format!("Text {i}")).collect();
        let labels = (0..10).map(|_| "A".to_string()).collect();

        let dataset = TextDataset::new(texts, labels).expect("Operation failed");
        let (train, test) = dataset
            .train_test_split(0.3, Some(42))
            .expect("Operation failed");

        assert_eq!(train.len(), 7);
        assert_eq!(test.len(), 3);
    }

    #[test]
    fn test_feature_selector() {
        let mut features = Array2::zeros((5, 3));
        features[[0, 0]] = 1.0;
        features[[1, 0]] = 1.0;
        features[[2, 0]] = 1.0;

        for i in 0..5 {
            features[[i, 1]] = 1.0;
        }

        features[[0, 2]] = 1.0;

        let mut selector = TextFeatureSelector::new()
            .set_min_df(0.25)
            .expect("Operation failed")
            .set_max_df(0.75)
            .expect("Operation failed");

        let filtered = selector.fit_transform(&features).expect("Operation failed");
        assert_eq!(filtered.ncols(), 1);
    }

    #[test]
    fn test_classification_metrics() {
        let predictions = vec![1_usize, 0, 1, 1, 0];
        let true_labels = vec![1_usize, 0, 1, 0, 0];

        let metrics = TextClassificationMetrics::new();
        let accuracy = metrics
            .accuracy(&predictions, &true_labels)
            .expect("Operation failed");
        assert_eq!(accuracy, 0.8);

        let (precision, recall, f1) = metrics
            .binary_metrics(&predictions, &true_labels)
            .expect("Operation failed");
        assert!((precision - 0.667).abs() < 0.001);
        assert_eq!(recall, 1.0);
        assert!((f1 - 0.8).abs() < 0.001);
    }

    // ─── Multinomial NB Tests ────────────────────────────────────────

    #[test]
    fn test_multinomial_nb_basic() {
        // Simple 2-class problem with 3 features
        let features = Array2::from_shape_vec(
            (6, 3),
            vec![
                3.0, 1.0, 0.0, // positive
                2.0, 2.0, 0.0, // positive
                4.0, 0.0, 1.0, // positive
                0.0, 1.0, 3.0, // negative
                0.0, 2.0, 2.0, // negative
                1.0, 0.0, 4.0, // negative
            ],
        )
        .expect("shape");

        let labels = vec![
            "pos".to_string(),
            "pos".to_string(),
            "pos".to_string(),
            "neg".to_string(),
            "neg".to_string(),
            "neg".to_string(),
        ];

        let mut clf = MultinomialNaiveBayes::new(1.0);
        clf.fit(&features, &labels).expect("fit");

        // Test with clearly positive sample
        let test = Array2::from_shape_vec((1, 3), vec![5.0, 0.0, 0.0]).expect("shape");
        let pred = clf.predict(&test).expect("predict");
        assert_eq!(pred[0], "pos");

        // Test with clearly negative sample
        let test = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 5.0]).expect("shape");
        let pred = clf.predict(&test).expect("predict");
        assert_eq!(pred[0], "neg");
    }

    // ─── Bernoulli NB Tests ──────────────────────────────────────────

    #[test]
    fn test_bernoulli_nb_basic() {
        let features = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 1.0, 0.0, 0.0, // pos
                1.0, 0.0, 1.0, 0.0, // pos
                0.0, 1.0, 1.0, 0.0, // pos
                0.0, 0.0, 0.0, 1.0, // neg
                0.0, 0.0, 1.0, 1.0, // neg
                0.0, 1.0, 0.0, 1.0, // neg
            ],
        )
        .expect("shape");

        let labels = vec![
            "pos".to_string(),
            "pos".to_string(),
            "pos".to_string(),
            "neg".to_string(),
            "neg".to_string(),
            "neg".to_string(),
        ];

        let mut clf = BernoulliNaiveBayes::new(1.0);
        clf.fit(&features, &labels).expect("fit");

        let test = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).expect("shape");
        let pred = clf.predict(&test).expect("predict");
        assert_eq!(pred[0], "pos");

        let test = Array2::from_shape_vec((1, 4), vec![0.0, 0.0, 0.0, 1.0]).expect("shape");
        let pred = clf.predict(&test).expect("predict");
        assert_eq!(pred[0], "neg");
    }

    // ─── TF-IDF Cosine Classifier Tests ──────────────────────────────

    #[test]
    fn test_tfidf_cosine_classifier() {
        let features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, // A
                0.9, 0.1, 0.0, // A
                0.0, 0.0, 1.0, // B
                0.1, 0.0, 0.9, // B
            ],
        )
        .expect("shape");

        let labels = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];

        let mut clf = TfidfCosineClassifier::new(1);
        clf.fit(&features, &labels).expect("fit");

        let test = Array2::from_shape_vec((1, 3), vec![0.8, 0.2, 0.0]).expect("shape");
        let pred = clf.predict(&test).expect("predict");
        assert_eq!(pred[0], "A");
    }

    // ─── Feature Hashing Tests ───────────────────────────────────────

    #[test]
    fn test_feature_hasher() {
        let hasher = FeatureHasher::new(100);

        let features = hasher.transform_text("the quick brown fox").expect("hash");
        assert_eq!(features.len(), 100);

        // Should have non-zero entries
        let nnz = features.iter().filter(|&&v| v != 0.0).count();
        assert!(nnz > 0);
    }

    #[test]
    fn test_feature_hasher_batch() {
        let hasher = FeatureHasher::new(50);

        let texts = vec!["hello world", "foo bar baz"];
        let matrix = hasher.transform_batch(&texts).expect("batch");

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 50);
    }

    #[test]
    fn test_feature_hasher_deterministic() {
        let hasher = FeatureHasher::new(100);

        let f1 = hasher.transform_text("hello world").expect("h1");
        let f2 = hasher.transform_text("hello world").expect("h2");

        for i in 0..100 {
            assert_eq!(f1[i], f2[i]);
        }
    }

    // ─── Multi-Label Tests ───────────────────────────────────────────

    #[test]
    fn test_multi_label_classifier() {
        let features = Array2::from_shape_vec(
            (4, 3),
            vec![
                3.0, 1.0, 0.0, // sports + positive
                2.0, 2.0, 0.0, // sports
                0.0, 1.0, 3.0, // tech + negative
                0.0, 0.0, 4.0, // tech
            ],
        )
        .expect("shape");

        let label_sets = vec![
            vec!["sports".to_string(), "positive".to_string()],
            vec!["sports".to_string()],
            vec!["tech".to_string(), "negative".to_string()],
            vec!["tech".to_string()],
        ];

        let mut clf = MultiLabelClassifier::new(0.5);
        clf.fit(&features, &label_sets).expect("fit");

        let test = Array2::from_shape_vec((1, 3), vec![4.0, 0.0, 0.0]).expect("shape");
        let preds = clf.predict(&test).expect("predict");
        assert!(!preds.is_empty());
        // Should predict sports-related labels
    }

    // ─── Cross-Validation Tests ──────────────────────────────────────

    #[test]
    fn test_cross_validation() {
        // Create a simple linearly separable dataset
        let n = 20;
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            if i < n / 2 {
                if j == 0 {
                    3.0
                } else {
                    0.0
                }
            } else {
                if j == 0 {
                    0.0
                } else {
                    3.0
                }
            }
        });

        let labels: Vec<String> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let result = cross_validate_nb(&features, &labels, 5, 1.0, Some(42)).expect("cv");

        assert_eq!(result.fold_results.len(), 5);
        // With linearly separable data, should get high accuracy
        assert!(
            result.mean_accuracy >= 0.5,
            "Mean accuracy: {}",
            result.mean_accuracy
        );
    }

    #[test]
    fn test_cross_validation_invalid_k() {
        let features = Array2::zeros((5, 2));
        let labels = vec!["A".to_string(); 5];

        let result = cross_validate_nb(&features, &labels, 1, 1.0, None);
        assert!(result.is_err());

        let result = cross_validate_nb(&features, &labels, 10, 1.0, None);
        assert!(result.is_err());
    }
}
