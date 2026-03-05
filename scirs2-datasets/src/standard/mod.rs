//! Standard machine learning datasets (embedded, no download required)
//!
//! Provides well-known benchmark datasets fully embedded in the binary.
//! Each loader returns a `DatasetResult` with data, target, feature names,
//! target names, and description -- following the scikit-learn convention.
//!
//! Available datasets:
//! - **Iris** (150 samples, 4 features, 3 classes)
//! - **Wine** (178 samples, 13 features, 3 classes)
//! - **Breast Cancer Wisconsin** (569 samples, 30 features, 2 classes)
//! - **Digits** (1797 samples, 64 features, 10 classes -- 8x8 images)
//! - **Boston Housing** (506 samples, 13 features, regression -- with deprecation note)

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};

/// Result type for standard dataset loaders
///
/// Contains all the metadata associated with a standard dataset, following
/// the scikit-learn `Bunch` convention.
#[derive(Debug, Clone)]
pub struct DatasetResult {
    /// Feature matrix (n_samples x n_features)
    pub data: Array2<f64>,
    /// Target array (n_samples,) -- class labels or regression targets
    pub target: Array1<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Target class names (for classification) or description (for regression)
    pub target_names: Vec<String>,
    /// Dataset description
    pub description: String,
}

impl DatasetResult {
    /// Number of samples in the dataset
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Number of features in the dataset
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Shape as (n_samples, n_features)
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples(), self.n_features())
    }
}

// ============================================================================
// Include embedded data modules
// ============================================================================

mod boston_data;
mod breast_cancer_data;
mod digits_data;
mod iris_data;
mod wine_data;

/// Load the Iris flower dataset (Fisher, 1936)
///
/// A classic multiclass classification dataset with 150 samples of iris flowers.
/// Each sample has 4 features (sepal/petal length/width) and belongs to one of
/// 3 species: Setosa, Versicolor, or Virginica.
///
/// # Returns
///
/// A `DatasetResult` with:
/// - `data`: (150, 4) feature matrix
/// - `target`: (150,) class labels {0, 1, 2}
/// - `feature_names`: sepal_length, sepal_width, petal_length, petal_width
/// - `target_names`: setosa, versicolor, virginica
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::standard::load_iris;
///
/// let iris = load_iris().expect("ok");
/// assert_eq!(iris.n_samples(), 150);
/// assert_eq!(iris.n_features(), 4);
/// assert_eq!(iris.target_names.len(), 3);
/// ```
pub fn load_iris() -> Result<DatasetResult> {
    iris_data::load()
}

/// Load the Wine recognition dataset (Aeberhard, Coomans & de Vel, 1992)
///
/// Chemical analysis of wines grown in the same region in Italy derived from
/// three different cultivars. 178 samples, 13 chemical features, 3 classes.
///
/// # Returns
///
/// A `DatasetResult` with:
/// - `data`: (178, 13) feature matrix
/// - `target`: (178,) class labels {0, 1, 2}
/// - `feature_names`: alcohol, malic_acid, ash, etc.
/// - `target_names`: class_0, class_1, class_2
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::standard::load_wine;
///
/// let wine = load_wine().expect("ok");
/// assert_eq!(wine.n_samples(), 178);
/// assert_eq!(wine.n_features(), 13);
/// ```
pub fn load_wine() -> Result<DatasetResult> {
    wine_data::load()
}

/// Load the Breast Cancer Wisconsin (Diagnostic) dataset
///
/// Features are computed from a digitized image of a fine needle aspirate (FNA)
/// of a breast mass. 569 samples, 30 features, 2 classes (malignant/benign).
///
/// # Returns
///
/// A `DatasetResult` with:
/// - `data`: (569, 30) feature matrix
/// - `target`: (569,) class labels {0=malignant, 1=benign}
/// - `feature_names`: 30 computed features (mean, SE, worst of radius/texture/etc.)
/// - `target_names`: malignant, benign
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::standard::load_breast_cancer;
///
/// let bc = load_breast_cancer().expect("ok");
/// assert_eq!(bc.n_samples(), 569);
/// assert_eq!(bc.n_features(), 30);
/// ```
pub fn load_breast_cancer() -> Result<DatasetResult> {
    breast_cancer_data::load()
}

/// Load the Digits dataset (8x8 handwritten digit images)
///
/// Each image is an 8x8 pixel grayscale image of a handwritten digit (0-9).
/// 1797 samples, 64 features (flattened pixels), 10 classes.
///
/// This is a generated approximation of the NIST Special Database 19,
/// using deterministic digit patterns with controlled noise.
///
/// # Returns
///
/// A `DatasetResult` with:
/// - `data`: (1797, 64) feature matrix with pixel values in [0, 16]
/// - `target`: (1797,) digit labels {0..9}
/// - `feature_names`: pixel_0 through pixel_63
/// - `target_names`: "0" through "9"
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::standard::load_digits;
///
/// let digits = load_digits().expect("ok");
/// assert_eq!(digits.n_samples(), 1797);
/// assert_eq!(digits.n_features(), 64);
/// ```
pub fn load_digits() -> Result<DatasetResult> {
    digits_data::load()
}

/// Load the Boston Housing dataset (Harrison & Rubinfeld, 1978)
///
/// **DEPRECATION NOTE**: This dataset has ethical concerns regarding the
/// variable "B" which was designed to capture racial demographics. Usage of
/// this dataset in research is discouraged. Consider using the California
/// Housing dataset instead.
///
/// 506 samples, 13 features, regression target (median home value in $1000s).
///
/// This version uses a deterministic synthetic approximation that preserves
/// the statistical properties of the original dataset.
///
/// # Returns
///
/// A `DatasetResult` with:
/// - `data`: (506, 13) feature matrix
/// - `target`: (506,) median home values
/// - `feature_names`: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
/// - `target_names`: ["MEDV"]
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::standard::load_boston;
///
/// let boston = load_boston().expect("ok");
/// assert_eq!(boston.n_samples(), 506);
/// assert_eq!(boston.n_features(), 13);
/// ```
pub fn load_boston() -> Result<DatasetResult> {
    boston_data::load()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iris_shape() {
        let ds = load_iris().expect("ok");
        assert_eq!(ds.n_samples(), 150);
        assert_eq!(ds.n_features(), 4);
        assert_eq!(ds.target.len(), 150);
        assert_eq!(ds.feature_names.len(), 4);
        assert_eq!(ds.target_names.len(), 3);
    }

    #[test]
    fn test_iris_labels() {
        let ds = load_iris().expect("ok");
        for &v in ds.target.iter() {
            assert!(v == 0.0 || v == 1.0 || v == 2.0, "Invalid iris label {v}");
        }
        // 50 of each class
        let count_0 = ds.target.iter().filter(|&&v| v == 0.0).count();
        let count_1 = ds.target.iter().filter(|&&v| v == 1.0).count();
        let count_2 = ds.target.iter().filter(|&&v| v == 2.0).count();
        assert_eq!(count_0, 50);
        assert_eq!(count_1, 50);
        assert_eq!(count_2, 50);
    }

    #[test]
    fn test_wine_shape() {
        let ds = load_wine().expect("ok");
        assert_eq!(ds.n_samples(), 178);
        assert_eq!(ds.n_features(), 13);
        assert_eq!(ds.target.len(), 178);
        assert_eq!(ds.feature_names.len(), 13);
        assert_eq!(ds.target_names.len(), 3);
    }

    #[test]
    fn test_wine_labels() {
        let ds = load_wine().expect("ok");
        for &v in ds.target.iter() {
            assert!(v == 0.0 || v == 1.0 || v == 2.0, "Invalid wine label {v}");
        }
    }

    #[test]
    fn test_breast_cancer_shape() {
        let ds = load_breast_cancer().expect("ok");
        assert_eq!(ds.n_samples(), 569);
        assert_eq!(ds.n_features(), 30);
        assert_eq!(ds.target.len(), 569);
        assert_eq!(ds.feature_names.len(), 30);
        assert_eq!(ds.target_names.len(), 2);
    }

    #[test]
    fn test_breast_cancer_labels() {
        let ds = load_breast_cancer().expect("ok");
        for &v in ds.target.iter() {
            assert!(v == 0.0 || v == 1.0, "Invalid breast cancer label {v}");
        }
    }

    #[test]
    fn test_digits_shape() {
        let ds = load_digits().expect("ok");
        assert_eq!(ds.n_samples(), 1797);
        assert_eq!(ds.n_features(), 64);
        assert_eq!(ds.target.len(), 1797);
        assert_eq!(ds.feature_names.len(), 64);
        assert_eq!(ds.target_names.len(), 10);
    }

    #[test]
    fn test_digits_labels() {
        let ds = load_digits().expect("ok");
        for &v in ds.target.iter() {
            assert!(
                v >= 0.0 && v <= 9.0 && v == v.floor(),
                "Invalid digit label {v}"
            );
        }
    }

    #[test]
    fn test_digits_pixel_range() {
        let ds = load_digits().expect("ok");
        for row in ds.data.rows() {
            for &v in row.iter() {
                assert!(
                    (0.0..=16.0).contains(&v),
                    "Pixel value {v} out of range [0, 16]"
                );
            }
        }
    }

    #[test]
    fn test_boston_shape() {
        let ds = load_boston().expect("ok");
        assert_eq!(ds.n_samples(), 506);
        assert_eq!(ds.n_features(), 13);
        assert_eq!(ds.target.len(), 506);
        assert_eq!(ds.feature_names.len(), 13);
    }

    #[test]
    fn test_boston_target_positive() {
        let ds = load_boston().expect("ok");
        for &v in ds.target.iter() {
            assert!(v > 0.0, "Boston target should be positive, got {v}");
        }
    }

    #[test]
    fn test_dataset_result_methods() {
        let ds = load_iris().expect("ok");
        assert_eq!(ds.shape(), (150, 4));
        assert!(!ds.description.is_empty());
    }

    #[test]
    fn test_all_datasets_consistent() {
        let datasets: Vec<(&str, DatasetResult)> = vec![
            ("iris", load_iris().expect("ok")),
            ("wine", load_wine().expect("ok")),
            ("breast_cancer", load_breast_cancer().expect("ok")),
            ("digits", load_digits().expect("ok")),
            ("boston", load_boston().expect("ok")),
        ];

        for (name, ds) in &datasets {
            assert_eq!(
                ds.data.nrows(),
                ds.target.len(),
                "{name}: data rows != target len"
            );
            assert_eq!(
                ds.data.ncols(),
                ds.feature_names.len(),
                "{name}: data cols != feature_names len"
            );
            assert!(
                !ds.description.is_empty(),
                "{name}: description should not be empty"
            );
        }
    }
}
