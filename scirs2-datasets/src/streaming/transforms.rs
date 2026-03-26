//! Lazy dataset transformations for streaming pipelines.
//!
//! Provides a composable [`Transform`] trait and concrete implementations
//! (`Normalize`, `Filter`, `MapFeatures`) that can be chained into a
//! [`TransformPipeline`].  All transforms operate on [`StreamingDataChunk`]
//! values produced by [`NewStreamingIterator`], enabling fully lazy, zero-
//! intermediate-copy processing.

use crate::error::DatasetsError;
use crate::streaming::iterator::{NewStreamingIterator, StreamingDataChunk};
use scirs2_core::ndarray::{Array1, Array2, Axis};

/// Type alias for a boxed row-level predicate used by [`Filter`].
type RowPredicate = Box<dyn Fn(&[f64]) -> bool + Send + Sync>;

/// Type alias for a boxed feature-mapping function used by [`MapFeatures`].
type FeatureMapFn = Box<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

/// A stateless (or internally-mutable) operation on a [`StreamingDataChunk`].
///
/// Implementing types must be `Send + Sync` so that pipelines can be safely
/// moved across threads.
pub trait Transform: Send + Sync {
    /// Apply the transformation to `chunk`, returning a (potentially new)
    /// chunk.  Implementations may mutate in place and return the same chunk,
    /// or allocate a new one.
    fn apply(&self, chunk: StreamingDataChunk) -> Result<StreamingDataChunk, DatasetsError>;
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------

/// Per-feature z-score normalisation: `x ← (x − mean) / std`.
///
/// Features with zero standard deviation are left unchanged (i.e. the
/// column remains as-is rather than becoming NaN).
#[derive(Debug, Clone)]
pub struct Normalize {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Normalize {
    /// Fit from a single `Array2<f64>` (all rows visible at once).
    pub fn fit(data: &Array2<f64>) -> Self {
        let mean_arr = data
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(data.ncols()));
        // Use sample standard deviation (ddof=1) to match sklearn/PyTorch convention
        let std_arr = data.std_axis(Axis(0), 1.0);
        Self {
            mean: mean_arr.to_vec(),
            std: std_arr.to_vec(),
        }
    }

    /// Incremental fit over all chunks produced by `iter`.
    ///
    /// Uses Welford's online algorithm to compute mean and variance in a
    /// single pass, consuming (then resetting) the iterator.
    pub fn fit_from_chunks(iter: &mut NewStreamingIterator) -> Result<Self, DatasetsError> {
        let nf = iter.n_features();
        if nf == 0 {
            return Ok(Self {
                mean: vec![],
                std: vec![],
            });
        }

        let mut count = 0usize;
        let mut mean = vec![0.0f64; nf];
        let mut m2 = vec![0.0f64; nf]; // sum of squared deviations

        for chunk_res in iter.by_ref() {
            let chunk = chunk_res?;
            for row in chunk.features.rows() {
                count += 1;
                for (j, &val) in row.iter().enumerate() {
                    let delta = val - mean[j];
                    mean[j] += delta / count as f64;
                    let delta2 = val - mean[j];
                    m2[j] += delta * delta2;
                }
            }
        }

        iter.reset();

        let std_dev: Vec<f64> = m2
            .into_iter()
            .map(|s| {
                if count > 1 {
                    (s / (count - 1) as f64).sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        Ok(Self { mean, std: std_dev })
    }

    /// Access fitted means (one per feature).
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Access fitted standard deviations (one per feature).
    pub fn std(&self) -> &[f64] {
        &self.std
    }
}

impl Transform for Normalize {
    fn apply(&self, mut chunk: StreamingDataChunk) -> Result<StreamingDataChunk, DatasetsError> {
        let nf = chunk.features.ncols();
        if nf != self.mean.len() {
            return Err(DatasetsError::InvalidFormat(format!(
                "Normalize: chunk has {nf} features, but was fitted on {}",
                self.mean.len()
            )));
        }
        for mut row in chunk.features.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                let s = self.std[j];
                if s > 0.0 {
                    *val = (*val - self.mean[j]) / s;
                }
            }
        }
        Ok(chunk)
    }
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

/// Row-level filter: keeps only rows for which `condition(&row) == true`.
pub struct Filter {
    condition: RowPredicate,
}

impl Filter {
    /// Create a filter from an arbitrary predicate on a row's feature slice.
    pub fn new(f: impl Fn(&[f64]) -> bool + Send + Sync + 'static) -> Self {
        Self {
            condition: Box::new(f),
        }
    }
}

impl Transform for Filter {
    fn apply(&self, chunk: StreamingDataChunk) -> Result<StreamingDataChunk, DatasetsError> {
        let nf = chunk.features.ncols();
        let n_rows = chunk.features.nrows();

        let mut keep_feat: Vec<f64> = Vec::new();
        let mut keep_labels: Vec<f64> = Vec::new();
        let mut kept = 0usize;

        for i in 0..n_rows {
            let row: Vec<f64> = chunk.features.row(i).to_vec();
            if (self.condition)(&row) {
                keep_feat.extend_from_slice(&row);
                if let Some(ref lbls) = chunk.labels {
                    keep_labels.push(if i < lbls.len() { lbls[i] } else { 0.0 });
                }
                kept += 1;
            }
        }

        let features = if kept == 0 {
            Array2::zeros((0, nf.max(1)))
        } else {
            Array2::from_shape_vec((kept, nf), keep_feat)
                .map_err(|e| DatasetsError::ComputationError(format!("Filter shape: {e}")))?
        };

        let labels = if chunk.labels.is_some() {
            Some(keep_labels)
        } else {
            None
        };

        Ok(StreamingDataChunk {
            features,
            labels,
            chunk_id: chunk.chunk_id,
        })
    }
}

// ---------------------------------------------------------------------------
// MapFeatures
// ---------------------------------------------------------------------------

/// Row-level feature mapping: applies a function `Array1<f64> → Array1<f64>`
/// to every row independently.
///
/// The output dimensionality may differ from the input; all rows must produce
/// the same output length.
pub struct MapFeatures {
    transform: FeatureMapFn,
}

impl MapFeatures {
    /// Create a feature map from an arbitrary function.
    pub fn new(f: impl Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static) -> Self {
        Self {
            transform: Box::new(f),
        }
    }
}

impl Transform for MapFeatures {
    fn apply(&self, chunk: StreamingDataChunk) -> Result<StreamingDataChunk, DatasetsError> {
        let n_rows = chunk.features.nrows();
        if n_rows == 0 {
            return Ok(chunk);
        }

        // Apply transform to the first row to discover output dimensionality
        let first_row = chunk.features.row(0).to_owned();
        let first_out = (self.transform)(&first_row);
        let out_nf = first_out.len();

        let mut out_flat: Vec<f64> = Vec::with_capacity(n_rows * out_nf);
        out_flat.extend(first_out.iter().copied());

        for i in 1..n_rows {
            let row = chunk.features.row(i).to_owned();
            let out = (self.transform)(&row);
            if out.len() != out_nf {
                return Err(DatasetsError::InvalidFormat(format!(
                    "MapFeatures: row {i} produced {} features, expected {out_nf}",
                    out.len()
                )));
            }
            out_flat.extend(out.iter().copied());
        }

        let features = Array2::from_shape_vec((n_rows, out_nf), out_flat)
            .map_err(|e| DatasetsError::ComputationError(format!("MapFeatures shape: {e}")))?;

        Ok(StreamingDataChunk {
            features,
            labels: chunk.labels,
            chunk_id: chunk.chunk_id,
        })
    }
}

// ---------------------------------------------------------------------------
// TransformPipeline
// ---------------------------------------------------------------------------

/// An ordered sequence of [`Transform`] steps applied in the order they were
/// added.
pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
}

impl TransformPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Append a transform step and return `self` (builder pattern).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, t: impl Transform + 'static) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    /// Apply all transforms in order to `chunk`.
    pub fn apply_chunk(
        &self,
        chunk: StreamingDataChunk,
    ) -> Result<StreamingDataChunk, DatasetsError> {
        let mut current = chunk;
        for transform in &self.transforms {
            current = transform.apply(current)?;
        }
        Ok(current)
    }

    /// Number of transforms in this pipeline.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns `true` if no transforms have been added.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::iterator::{DataSource, NewStreamingIterator, StreamingIteratorConfig};
    use scirs2_core::ndarray::Array2;

    fn make_chunk(data: Vec<Vec<f64>>) -> StreamingDataChunk {
        let n = data.len();
        let f = if n == 0 { 1 } else { data[0].len() };
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        StreamingDataChunk {
            features: Array2::from_shape_vec((n, f), flat).expect("shape"),
            labels: None,
            chunk_id: 0,
        }
    }

    #[test]
    fn test_normalize_transform() {
        // Build data with known mean/std
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let arr =
            Array2::from_shape_vec((3, 2), data.iter().flatten().copied().collect::<Vec<_>>())
                .unwrap();
        let norm = Normalize::fit(&arr);

        let chunk = make_chunk(data);
        let out = norm.apply(chunk).expect("normalize");

        // After normalisation the column means should be ≈ 0 and stds ≈ 1
        let col0_mean: f64 = out.features.column(0).mean().unwrap_or(0.0);
        let col1_mean: f64 = out.features.column(1).mean().unwrap_or(0.0);
        assert!(col0_mean.abs() < 1e-10, "col0 mean {col0_mean}");
        assert!(col1_mean.abs() < 1e-10, "col1 mean {col1_mean}");

        let col0_std = out.features.column(0).std(1.0);
        assert!((col0_std - 1.0).abs() < 1e-10, "col0 std {col0_std}");
    }

    #[test]
    fn test_filter_transform() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let chunk = make_chunk(data);
        // Keep rows where first feature > 2
        let filter = Filter::new(|row| row[0] > 2.0);
        let out = filter.apply(chunk).expect("filter");
        assert_eq!(out.n_rows(), 3);
        assert!(out.features.column(0).iter().all(|&v| v > 2.0));
    }

    #[test]
    fn test_filter_all_removed() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let chunk = make_chunk(data);
        let filter = Filter::new(|row| row[0] > 100.0);
        let out = filter.apply(chunk).expect("filter");
        assert_eq!(out.n_rows(), 0);
    }

    #[test]
    fn test_map_features_double() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let chunk = make_chunk(data);
        let map = MapFeatures::new(|row| row.mapv(|x| x * 2.0));
        let out = map.apply(chunk).expect("map");
        assert_eq!(out.features[[0, 0]], 2.0);
        assert_eq!(out.features[[0, 1]], 4.0);
        assert_eq!(out.features[[1, 0]], 6.0);
    }

    #[test]
    fn test_transform_pipeline() {
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let arr =
            Array2::from_shape_vec((10, 2), rows.iter().flatten().copied().collect::<Vec<_>>())
                .unwrap();
        let norm = Normalize::fit(&arr);

        // Pipeline: normalise → filter out rows with col0 < 0 → double values
        let pipeline = TransformPipeline::new()
            .add(norm)
            .add(Filter::new(|row| row[0] >= -0.5))
            .add(MapFeatures::new(|row| row.mapv(|x| x * 2.0)));

        assert_eq!(pipeline.len(), 3);

        let chunk = make_chunk(rows);
        let out = pipeline.apply_chunk(chunk).expect("pipeline");
        // After normalisation + filter, some rows should remain
        assert!(out.n_rows() > 0);
    }

    #[test]
    fn test_normalize_fit_from_chunks() {
        let rows: Vec<Vec<f64>> = (0..30_usize)
            .map(|i| vec![(i % 10) as f64, ((i % 5) * 2) as f64])
            .collect();
        let config = StreamingIteratorConfig {
            chunk_size: 10,
            ..Default::default()
        };
        let mut iter =
            NewStreamingIterator::new(DataSource::InMemory(rows.clone()), config).expect("iter");
        let norm = Normalize::fit_from_chunks(&mut iter).expect("fit");

        // Check mean is correct (should match the data's column means)
        let expected_mean0: f64 = rows.iter().map(|r| r[0]).sum::<f64>() / rows.len() as f64;
        assert!((norm.mean()[0] - expected_mean0).abs() < 1e-10);
        // std should be positive
        assert!(norm.std()[0] > 0.0);
        assert!(norm.std()[1] > 0.0);
    }

    #[test]
    fn test_pipeline_empty_chunk() {
        let chunk = StreamingDataChunk {
            features: Array2::zeros((0, 3)),
            labels: None,
            chunk_id: 0,
        };
        let map = MapFeatures::new(|row| row.mapv(|x| x + 1.0));
        let out = map.apply(chunk).expect("map empty");
        assert_eq!(out.n_rows(), 0);
    }
}
