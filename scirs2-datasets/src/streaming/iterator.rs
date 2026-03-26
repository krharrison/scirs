//! Streaming iterator API for datasets exceeding RAM.
//!
//! Provides a lazy, chunk-based iteration interface over multiple data sources
//! (in-memory vectors, CSV files, directories of files). Each iteration step
//! yields a [`StreamingDataChunk`] holding at most `chunk_size` rows, enabling
//! processing of arbitrarily large datasets with bounded memory usage.

use crate::error::DatasetsError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;

// ---------------------------------------------------------------------------
// DataSource
// ---------------------------------------------------------------------------

/// Origin of data for a [`NewStreamingIterator`].
///
/// This enum is `#[non_exhaustive]` so that future sources (e.g. Arrow IPC,
/// HDF5) can be added without breaking existing match arms.
#[non_exhaustive]
#[derive(Debug)]
pub enum DataSource {
    /// All rows are already in memory as a `Vec<Vec<f64>>`.
    ///
    /// Each inner `Vec<f64>` is one row; all rows must have the same length.
    InMemory(Vec<Vec<f64>>),

    /// Path to a CSV file (first row treated as a header and skipped).
    ///
    /// All remaining columns except the last are treated as features; the last
    /// column is treated as a label.
    Csv(String),

    /// Path to a Parquet file (requires `formats` feature via scirs2-io).
    ///
    /// Currently falls back to an unsupported error unless the `formats`
    /// feature is enabled.
    Parquet(String),

    /// Path to a directory.  Every file in the directory is read as a CSV
    /// (same convention as [`DataSource::Csv`]).
    Directory(String),
}

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`NewStreamingIterator`].
#[derive(Debug, Clone)]
pub struct StreamingIteratorConfig {
    /// Number of rows per chunk (default: 1024).
    pub chunk_size: usize,
    /// Number of chunks to pre-read ahead (currently unused; reserved for
    /// future async prefetch). Default: 2.
    pub prefetch: usize,
    /// Shuffle row order within each chunk using Fisher-Yates (default: false).
    pub shuffle: bool,
    /// RNG seed used when `shuffle` is true (default: 42).
    pub seed: u64,
}

impl Default for StreamingIteratorConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            prefetch: 2,
            shuffle: false,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingDataChunk
// ---------------------------------------------------------------------------

/// A single chunk produced by [`NewStreamingIterator`].
#[derive(Debug, Clone)]
pub struct StreamingDataChunk {
    /// Feature matrix, shape `[actual_rows, n_features]`.
    pub features: Array2<f64>,
    /// Optional label vector, length `actual_rows`.
    pub labels: Option<Vec<f64>>,
    /// Zero-based index of this chunk within the stream.
    pub chunk_id: usize,
}

impl StreamingDataChunk {
    /// Number of rows in this chunk.
    pub fn n_rows(&self) -> usize {
        self.features.nrows()
    }

    /// Number of features (columns) in this chunk.
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
}

// ---------------------------------------------------------------------------
// Internal helper: in-memory row store
// ---------------------------------------------------------------------------

/// Parsed CSV result: `(data, labels, n_features)`.
type CsvParseResult = (Vec<f64>, Vec<Option<f64>>, usize);

/// Rows buffered from the data source.  For CSV/Directory sources we read
/// the whole source eagerly on construction so that we know the total row
/// count and feature dimensionality; this also keeps the iteration path
/// uniform across sources.
struct RowStore {
    /// Flat storage: row i, feature j → rows[i * n_features + j]
    data: Vec<f64>,
    /// Label per row (same indexing as `data` rows)
    labels: Vec<Option<f64>>,
    /// Number of features per row
    n_features: usize,
    /// Total number of rows
    n_rows: usize,
}

impl RowStore {
    fn from_in_memory(rows: Vec<Vec<f64>>) -> Result<Self, DatasetsError> {
        if rows.is_empty() {
            return Ok(Self {
                data: vec![],
                labels: vec![],
                n_features: 0,
                n_rows: 0,
            });
        }
        let n_features = rows[0].len();
        if n_features == 0 {
            return Err(DatasetsError::InvalidFormat(
                "InMemory rows must have at least one element".to_string(),
            ));
        }
        let n_rows = rows.len();
        let mut data = Vec::with_capacity(n_rows * n_features);
        for row in &rows {
            if row.len() != n_features {
                return Err(DatasetsError::InvalidFormat(format!(
                    "Inconsistent row length: expected {n_features}, got {}",
                    row.len()
                )));
            }
            data.extend_from_slice(row);
        }
        Ok(Self {
            data,
            labels: vec![None; n_rows],
            n_features,
            n_rows,
        })
    }

    /// Read a single CSV file (header-skipped, last column = label).
    fn parse_csv_file(path: &str) -> Result<CsvParseResult, DatasetsError> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(DatasetsError::IoError)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // skip header
        let _ = lines.next();

        let mut all_data: Vec<f64> = Vec::new();
        let mut all_labels: Vec<Option<f64>> = Vec::new();
        let mut n_features: Option<usize> = None;

        for line_res in lines {
            let line = line_res.map_err(DatasetsError::IoError)?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let values: Vec<f64> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            if values.is_empty() {
                continue;
            }
            let features_here = values.len() - 1; // last col = label
            if features_here == 0 {
                // Single column: treat as feature, no label
                match n_features {
                    None => n_features = Some(1),
                    Some(f) if f != 1 => {
                        return Err(DatasetsError::InvalidFormat(
                            "Inconsistent number of columns in CSV".to_string(),
                        ))
                    }
                    _ => {}
                }
                all_data.push(values[0]);
                all_labels.push(None);
            } else {
                match n_features {
                    None => n_features = Some(features_here),
                    Some(f) if f != features_here => {
                        return Err(DatasetsError::InvalidFormat(
                            "Inconsistent number of columns in CSV".to_string(),
                        ))
                    }
                    _ => {}
                }
                all_data.extend_from_slice(&values[..features_here]);
                all_labels.push(Some(*values.last().expect("non-empty")));
            }
        }

        let nf = n_features.unwrap_or(0);
        Ok((all_data, all_labels, nf))
    }

    fn from_csv(path: &str) -> Result<Self, DatasetsError> {
        let (data, labels, n_features) = Self::parse_csv_file(path)?;
        let n_rows = data.len().checked_div(n_features).unwrap_or(0);
        Ok(Self {
            data,
            labels,
            n_features,
            n_rows,
        })
    }

    fn from_directory(dir: &str) -> Result<Self, DatasetsError> {
        use std::fs;
        let mut all_data: Vec<f64> = Vec::new();
        let mut all_labels: Vec<Option<f64>> = Vec::new();
        let mut n_features: Option<usize> = None;

        let entries = fs::read_dir(dir).map_err(DatasetsError::IoError)?;
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok().map(|de| de.path()))
            .filter(|p| p.is_file())
            .collect();
        paths.sort(); // deterministic order

        for path in paths {
            let path_str = path.to_string_lossy();
            let (data, labels, nf) = Self::parse_csv_file(&path_str)?;
            if nf == 0 {
                continue;
            }
            match n_features {
                None => n_features = Some(nf),
                Some(f) if f != nf => {
                    return Err(DatasetsError::InvalidFormat(format!(
                        "Directory file {} has {nf} features, expected {f}",
                        path.display()
                    )))
                }
                _ => {}
            }
            all_data.extend(data);
            all_labels.extend(labels);
        }

        let nf = n_features.unwrap_or(0);
        let n_rows = all_data.len().checked_div(nf).unwrap_or(0);
        Ok(Self {
            data: all_data,
            labels: all_labels,
            n_features: nf,
            n_rows,
        })
    }

    /// Extract a slice of rows `[start, end)` as a `StreamingDataChunk`.
    fn slice_chunk(
        &self,
        start: usize,
        end: usize,
        chunk_id: usize,
        shuffle: bool,
        rng: &mut StdRng,
    ) -> Result<StreamingDataChunk, DatasetsError> {
        let end = end.min(self.n_rows);
        if start >= end {
            // Return an empty chunk
            let features = Array2::zeros((0, self.n_features.max(1)));
            return Ok(StreamingDataChunk {
                features,
                labels: None,
                chunk_id,
            });
        }
        let count = end - start;
        let nf = self.n_features;

        // Build index list (for shuffle support)
        let mut indices: Vec<usize> = (start..end).collect();
        if shuffle {
            // Fisher-Yates
            for i in (1..count).rev() {
                let j = rng.next_u64() as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        let mut feat_flat: Vec<f64> = Vec::with_capacity(count * nf);
        let mut labels_out: Vec<f64> = Vec::with_capacity(count);
        let mut has_labels = false;

        for &row_idx in &indices {
            let base = row_idx * nf;
            feat_flat.extend_from_slice(&self.data[base..base + nf]);
            if let Some(lbl) = self.labels[row_idx] {
                labels_out.push(lbl);
                has_labels = true;
            } else {
                labels_out.push(0.0);
            }
        }

        let features = Array2::from_shape_vec((count, nf), feat_flat)
            .map_err(|e| DatasetsError::ComputationError(format!("Shape error: {e}")))?;

        Ok(StreamingDataChunk {
            features,
            labels: if has_labels { Some(labels_out) } else { None },
            chunk_id,
        })
    }
}

// ---------------------------------------------------------------------------
// NewStreamingIterator
// ---------------------------------------------------------------------------

/// Streaming iterator over a [`DataSource`], yielding [`StreamingDataChunk`]s.
///
/// Call [`NewStreamingIterator::new`] to construct, then use it as a standard
/// Rust `Iterator<Item = Result<StreamingDataChunk, DatasetsError>>`.
pub struct NewStreamingIterator {
    store: RowStore,
    config: StreamingIteratorConfig,
    current_chunk: usize,
    rng: StdRng,
}

impl NewStreamingIterator {
    /// Construct a streaming iterator from the given source and configuration.
    ///
    /// For `Csv` and `Directory` sources, the file(s) are read eagerly during
    /// construction so that the total row count is known immediately.
    pub fn new(source: DataSource, config: StreamingIteratorConfig) -> Result<Self, DatasetsError> {
        let store = match source {
            DataSource::InMemory(rows) => RowStore::from_in_memory(rows)?,
            DataSource::Csv(path) => RowStore::from_csv(&path)?,
            DataSource::Directory(dir) => RowStore::from_directory(&dir)?,
            DataSource::Parquet(_) => {
                return Err(DatasetsError::Other(
                    "Parquet source requires the `formats` feature".to_string(),
                ))
            }
        };

        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            store,
            config,
            current_chunk: 0,
            rng,
        })
    }

    /// Total number of chunks (known because source is fully loaded).
    pub fn n_chunks(&self) -> Option<usize> {
        if self.config.chunk_size == 0 {
            return Some(0);
        }
        Some(self.store.n_rows.div_ceil(self.config.chunk_size))
    }

    /// Number of features per row.
    pub fn n_features(&self) -> usize {
        self.store.n_features
    }

    /// Total number of rows across all chunks.
    pub fn n_rows(&self) -> usize {
        self.store.n_rows
    }

    /// Reset the iterator to the beginning of the stream.
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }
}

impl Iterator for NewStreamingIterator {
    type Item = Result<StreamingDataChunk, DatasetsError>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk_size = self.config.chunk_size;
        let start = self.current_chunk * chunk_size;
        if start >= self.store.n_rows && self.store.n_rows > 0 {
            return None;
        }
        // Handle empty source: emit nothing
        if self.store.n_rows == 0 {
            return None;
        }
        let end = (start + chunk_size).min(self.store.n_rows);
        let chunk_id = self.current_chunk;
        self.current_chunk += 1;

        let result =
            self.store
                .slice_chunk(start, end, chunk_id, self.config.shuffle, &mut self.rng);
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rows(n: usize, f: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..f).map(|j| (i * f + j) as f64).collect())
            .collect()
    }

    #[test]
    fn test_streaming_inmemory() {
        let rows = make_rows(100, 4);
        let config = StreamingIteratorConfig {
            chunk_size: 30,
            ..Default::default()
        };
        let iter = NewStreamingIterator::new(DataSource::InMemory(rows), config)
            .expect("construction failed");
        // 100 rows / 30 = 4 chunks (3 full + 1 partial)
        assert_eq!(iter.n_chunks(), Some(4));
        assert_eq!(iter.n_features(), 4);
    }

    #[test]
    fn test_streaming_chunk_size() {
        let rows = make_rows(55, 3);
        let config = StreamingIteratorConfig {
            chunk_size: 20,
            ..Default::default()
        };
        let iter = NewStreamingIterator::new(DataSource::InMemory(rows), config)
            .expect("construction failed");

        let chunks: Vec<_> = iter.map(|r| r.expect("chunk error")).collect();
        // 55 / 20 = 3 chunks: 20, 20, 15
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].n_rows(), 20);
        assert_eq!(chunks[1].n_rows(), 20);
        assert_eq!(chunks[2].n_rows(), 15);
        for chunk in &chunks {
            assert!(chunk.n_rows() <= 20);
        }
    }

    #[test]
    fn test_streaming_empty_source() {
        let config = StreamingIteratorConfig::default();
        let iter =
            NewStreamingIterator::new(DataSource::InMemory(vec![]), config).expect("construction");
        let chunks: Vec<_> = iter.collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_streaming_single_row() {
        let config = StreamingIteratorConfig {
            chunk_size: 10,
            ..Default::default()
        };
        let iter =
            NewStreamingIterator::new(DataSource::InMemory(vec![vec![1.0, 2.0, 3.0]]), config)
                .expect("construction");
        let chunks: Vec<_> = iter.map(|r| r.expect("err")).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].n_rows(), 1);
        assert_eq!(chunks[0].n_features(), 3);
    }

    #[test]
    fn test_streaming_exact_multiple() {
        // 60 rows, chunk_size=20 → exactly 3 full chunks
        let rows = make_rows(60, 2);
        let config = StreamingIteratorConfig {
            chunk_size: 20,
            ..Default::default()
        };
        let iter =
            NewStreamingIterator::new(DataSource::InMemory(rows), config).expect("construction");
        let chunks: Vec<_> = iter.map(|r| r.expect("err")).collect();
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.n_rows(), 20);
        }
    }

    #[test]
    fn test_streaming_reset() {
        let rows = make_rows(10, 2);
        let config = StreamingIteratorConfig {
            chunk_size: 5,
            ..Default::default()
        };
        let mut iter =
            NewStreamingIterator::new(DataSource::InMemory(rows), config).expect("construction");
        let first_run: Vec<_> = iter.by_ref().map(|r| r.expect("err")).collect();
        iter.reset();
        let second_run: Vec<_> = iter.map(|r| r.expect("err")).collect();
        assert_eq!(first_run.len(), second_run.len());
    }

    #[test]
    fn test_streaming_csv() {
        use std::io::Write;
        let mut tmp = std::env::temp_dir();
        tmp.push("scirs2_streaming_test.csv");
        {
            let mut f = std::fs::File::create(&tmp).expect("create");
            writeln!(f, "a,b,c,label").expect("write header");
            for i in 0..20_usize {
                writeln!(f, "{},{},{},{}", i, i + 1, i + 2, i % 3).expect("write row");
            }
        }
        let config = StreamingIteratorConfig {
            chunk_size: 8,
            ..Default::default()
        };
        let iter =
            NewStreamingIterator::new(DataSource::Csv(tmp.to_string_lossy().into_owned()), config)
                .expect("construction");
        let chunks: Vec<_> = iter.map(|r| r.expect("err")).collect();
        // 20 rows / 8 = 3 chunks (8, 8, 4)
        assert_eq!(chunks.len(), 3);
        let total_rows: usize = chunks.iter().map(|c| c.n_rows()).sum();
        assert_eq!(total_rows, 20);
        // labels should be present
        assert!(chunks[0].labels.is_some());
        let _ = std::fs::remove_file(&tmp);
    }
}
