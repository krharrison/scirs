# Migrating from scikit-learn

This guide maps common scikit-learn workflows to their SciRS2 equivalents.
SciRS2 distributes ML functionality across several domain-specific crates
rather than a single monolithic package.

## Crate Mapping

| scikit-learn | SciRS2 Crate |
|-------------|--------------|
| `sklearn.cluster` | `scirs2-cluster` |
| `sklearn.decomposition` | `scirs2-stats` (PCA), `scirs2-linalg` (SVD) |
| `sklearn.metrics` | `scirs2-metrics` |
| `sklearn.preprocessing` | `scirs2-transform` |
| `sklearn.linear_model` | `scirs2-stats::regression` |
| `sklearn.neural_network` | `scirs2-neural` |
| `sklearn.datasets` | `scirs2-datasets` |

## Clustering

### K-Means

```rust,ignore
use scirs2_cluster::{KMeans, KMeansOptions};

let options = KMeansOptions::default()
    .with_n_clusters(3)
    .with_max_iter(300)
    .with_n_init(10);

let model = KMeans::new(options)?;
let labels = model.fit_predict(&data)?;
let centroids = model.cluster_centers();
let inertia = model.inertia();
```

### DBSCAN

```rust,ignore
use scirs2_cluster::{DBSCAN, DBSCANOptions};

let model = DBSCAN::new(DBSCANOptions {
    eps: 0.5,
    min_samples: 5,
})?;
let labels = model.fit_predict(&data)?;
// -1 indicates noise points
```

### Community Detection (Graph Clustering)

```rust,ignore
use scirs2_cluster::overlapping::{BigCLAM, DEMON};

// BigCLAM for overlapping communities
let model = BigCLAM::new(num_communities)?;
let memberships = model.fit(&adjacency_matrix)?;

// DEMON for ego-network based detection
let communities = DEMON::new(epsilon, min_community_size)?.detect(&graph)?;
```

## Dimensionality Reduction

### PCA

```rust,ignore
use scirs2_stats::multivariate::PCA;

let pca = PCA::new(n_components)?;
let reduced = pca.fit_transform(&data)?;

println!("Explained variance ratio: {:?}", pca.explained_variance_ratio());
println!("Components shape: {:?}", pca.components().shape());

// Transform new data
let new_reduced = pca.transform(&new_data)?;
```

## Regression

### Linear Regression

| scikit-learn | SciRS2 |
|-------------|--------|
| `LinearRegression().fit(X, y)` | `linear_regression(&x.view(), &y.view())?` |
| `Ridge(alpha=1.0).fit(X, y)` | `ridge(&x.view(), &y.view(), 1.0)?` |
| `Lasso(alpha=0.1).fit(X, y)` | `lasso(&x.view(), &y.view(), 0.1)?` |
| `ElasticNet(alpha=0.1, l1_ratio=0.5)` | `elastic_net(&x.view(), &y.view(), 0.1, 0.5)?` |

```rust,ignore
use scirs2_stats::regression::{linear_regression, ridge};

let result = linear_regression(&x.view(), &y.view())?;
println!("R^2: {:.4}", result.r_squared);
println!("Coefficients: {:?}", result.coefficients);

// Predictions
let y_pred = result.predict(&x_test.view())?;
```

## Metrics

### Classification Metrics

```rust,ignore
use scirs2_metrics::{accuracy, precision_recall_f1, confusion_matrix, roc_auc};

let acc = accuracy(&y_true, &y_pred)?;
let (precision, recall, f1) = precision_recall_f1(&y_true, &y_pred, "macro")?;
let cm = confusion_matrix(&y_true, &y_pred, num_classes)?;
let auc = roc_auc(&y_true, &y_scores)?;
```

### Calibration

```rust,ignore
use scirs2_metrics::calibration::{expected_calibration_error, reliability_diagram};

let ece = expected_calibration_error(&y_true, &y_prob, 15)?;
let (bin_confidences, bin_accuracies) = reliability_diagram(&y_true, &y_prob, 15)?;
```

## Preprocessing / Transforms

### Standardization

```rust,ignore
use scirs2_transform::{StandardScaler, MinMaxScaler};

let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&data)?;

// Transform new data with learned parameters
let new_scaled = scaler.transform(&new_data)?;

// Inverse transform
let original = scaler.inverse_transform(&scaled)?;
```

## Datasets

### Synthetic Generators

```rust,ignore
use scirs2_datasets::{make_classification, make_regression, make_blobs};

// Classification dataset
let (x, y) = make_classification(
    n_samples, n_features, n_informative, n_classes
)?;

// Regression dataset
let (x, y) = make_regression(n_samples, n_features, noise)?;

// Blob clusters
let (x, y) = make_blobs(n_samples, n_features, centers)?;
```

## Common Workflow Translation

Python:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = Ridge(alpha=1.0).fit(X_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
print(f"R^2: {r2_score(y_test, y_pred):.4f}")
```

Rust:
```rust,ignore
use scirs2_transform::StandardScaler;
use scirs2_stats::regression::ridge;

let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x_train)?;
let model = ridge(&x_scaled.view(), &y_train.view(), 1.0)?;
let x_test_scaled = scaler.transform(&x_test)?;
let y_pred = model.predict(&x_test_scaled.view())?;
println!("R^2: {:.4}", model.r_squared);
```
