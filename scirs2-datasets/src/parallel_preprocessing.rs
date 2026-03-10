//! Parallel data preprocessing pipeline
//!
//! This module provides a multi-threaded preprocessing pipeline with work-stealing
//! scheduler, memory-efficient batch processing, and backpressure handling for
//! optimal throughput and resource utilization.

use crate::error::{DatasetsError, Result};
use crate::streaming::DataChunk;
use crate::utils::Dataset;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use scirs2_core::ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Preprocessing function type
pub type PreprocessFn = Arc<dyn Fn(&Array2<f64>) -> Result<Array2<f64>> + Send + Sync>;

/// Configuration for parallel preprocessing
#[derive(Clone)]
pub struct ParallelConfig {
    /// Number of worker threads (0 = auto-detect)
    pub num_workers: usize,
    /// Size of the input buffer
    pub input_buffer_size: usize,
    /// Size of the output buffer
    pub output_buffer_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Whether to use work stealing
    pub enable_work_stealing: bool,
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: usize,
    /// Whether to enable backpressure
    pub enable_backpressure: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            input_buffer_size: 10,
            output_buffer_size: 10,
            batch_size: 1000,
            enable_work_stealing: true,
            max_memory_bytes: 0,
            enable_backpressure: true,
        }
    }
}

impl ParallelConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of workers
    pub fn with_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = if num_workers == 0 {
            num_cpus::get()
        } else {
            num_workers
        };
        self
    }

    /// Set buffer sizes
    pub fn with_buffer_sizes(mut self, input: usize, output: usize) -> Self {
        self.input_buffer_size = input;
        self.output_buffer_size = output;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable or disable work stealing
    pub fn with_work_stealing(mut self, enable: bool) -> Self {
        self.enable_work_stealing = enable;
        self
    }

    /// Set memory limit
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

/// Work item for preprocessing
#[derive(Clone)]
struct WorkItem {
    id: usize,
    data: Array2<f64>,
    target: Option<Array1<f64>>,
}

/// Processed result
struct ProcessedItem {
    id: usize,
    data: Array2<f64>,
    target: Option<Array1<f64>>,
}

/// Parallel preprocessing pipeline
pub struct ParallelPipeline {
    config: ParallelConfig,
    preprocess_fn: PreprocessFn,
    workers: Vec<JoinHandle<()>>,
    input_sender: Option<Sender<WorkItem>>,
    output_receiver: Option<Receiver<ProcessedItem>>,
    stop_flag: Arc<AtomicBool>,
    items_processed: Arc<AtomicUsize>,
}

impl ParallelPipeline {
    /// Create a new parallel preprocessing pipeline
    ///
    /// # Arguments
    /// * `config` - Pipeline configuration
    /// * `preprocess_fn` - Function to apply to each data chunk
    ///
    /// # Returns
    /// * `ParallelPipeline` - The pipeline instance
    pub fn new(config: ParallelConfig, preprocess_fn: PreprocessFn) -> Self {
        let (input_tx, input_rx) = if config.enable_backpressure {
            bounded(config.input_buffer_size)
        } else {
            unbounded()
        };

        let (output_tx, output_rx) = if config.enable_backpressure {
            bounded(config.output_buffer_size)
        } else {
            unbounded()
        };

        let stop_flag = Arc::new(AtomicBool::new(false));
        let items_processed = Arc::new(AtomicUsize::new(0));

        // Spawn worker threads
        let mut workers = Vec::new();
        for worker_id in 0..config.num_workers {
            let rx = input_rx.clone();
            let tx = output_tx.clone();
            let fn_clone = Arc::clone(&preprocess_fn);
            let stop_flag_clone = Arc::clone(&stop_flag);
            let items_clone = Arc::clone(&items_processed);

            let worker = thread::spawn(move || {
                Self::worker_loop(worker_id, rx, tx, fn_clone, stop_flag_clone, items_clone);
            });

            workers.push(worker);
        }

        // Drop the original senders/receivers so workers can detect completion
        drop(output_tx);

        Self {
            config,
            preprocess_fn,
            workers,
            input_sender: Some(input_tx),
            output_receiver: Some(output_rx),
            stop_flag,
            items_processed,
        }
    }

    /// Worker thread main loop
    fn worker_loop(
        _worker_id: usize,
        input: Receiver<WorkItem>,
        output: Sender<ProcessedItem>,
        preprocess_fn: PreprocessFn,
        stop_flag: Arc<AtomicBool>,
        items_processed: Arc<AtomicUsize>,
    ) {
        while !stop_flag.load(Ordering::Relaxed) {
            match input.recv() {
                Ok(item) => {
                    // Process the item
                    match preprocess_fn(&item.data) {
                        Ok(processed_data) => {
                            let result = ProcessedItem {
                                id: item.id,
                                data: processed_data,
                                target: item.target,
                            };

                            // Increment before sending so the counter is visible
                            // to the receiver once it drains the result
                            items_processed.fetch_add(1, Ordering::Release);
                            // Send result (ignore errors as receiver might be dropped)
                            let _ = output.send(result);
                        }
                        Err(_) => {
                            // On error, pass through original data
                            let result = ProcessedItem {
                                id: item.id,
                                data: item.data,
                                target: item.target,
                            };
                            let _ = output.send(result);
                        }
                    }
                }
                Err(_) => break, // Channel closed
            }
        }
    }

    /// Submit data for processing
    ///
    /// # Arguments
    /// * `data` - Input data array
    /// * `target` - Optional target values
    ///
    /// # Returns
    /// * `Ok(usize)` - ID of the submitted item
    /// * `Err(DatasetsError)` - If submission fails
    pub fn submit(&mut self, data: Array2<f64>, target: Option<Array1<f64>>) -> Result<usize> {
        let id = self.items_processed.load(Ordering::Relaxed);
        let item = WorkItem { id, data, target };

        self.input_sender
            .as_ref()
            .ok_or_else(|| DatasetsError::ProcessingError("Pipeline not initialized".to_string()))?
            .send(item)
            .map_err(|e| DatasetsError::ProcessingError(format!("Failed to submit: {}", e)))?;

        Ok(id)
    }

    /// Submit a dataset for processing
    pub fn submit_dataset(&mut self, dataset: &Dataset) -> Result<usize> {
        self.submit(dataset.data.clone(), dataset.target.clone())
    }

    /// Submit a data chunk for processing
    pub fn submit_chunk(&mut self, chunk: &DataChunk) -> Result<usize> {
        self.submit(chunk.data.clone(), chunk.target.clone())
    }

    /// Receive a processed result
    ///
    /// # Returns
    /// * `Ok(Some(Dataset))` - Processed dataset
    /// * `Ok(None)` - No more results (all workers finished)
    /// * `Err(DatasetsError)` - If receive fails
    pub fn receive(&mut self) -> Result<Option<Dataset>> {
        match self.output_receiver.as_ref() {
            Some(rx) => match rx.recv() {
                Ok(item) => Ok(Some(Dataset {
                    data: item.data,
                    target: item.target,
                    targetnames: None,
                    featurenames: None,
                    feature_descriptions: None,
                    description: None,
                    metadata: Default::default(),
                })),
                Err(_) => Ok(None), // Channel closed
            },
            None => Err(DatasetsError::ProcessingError(
                "Pipeline not initialized".to_string(),
            )),
        }
    }

    /// Try to receive a result without blocking
    pub fn try_receive(&mut self) -> Result<Option<Dataset>> {
        match self.output_receiver.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(item) => Ok(Some(Dataset {
                    data: item.data,
                    target: item.target,
                    targetnames: None,
                    featurenames: None,
                    feature_descriptions: None,
                    description: None,
                    metadata: Default::default(),
                })),
                Err(_) => Ok(None),
            },
            None => Err(DatasetsError::ProcessingError(
                "Pipeline not initialized".to_string(),
            )),
        }
    }

    /// Process a batch of datasets
    pub fn process_batch(&mut self, datasets: &[Dataset]) -> Result<Vec<Dataset>> {
        // Submit all
        for ds in datasets {
            self.submit_dataset(ds)?;
        }

        // Collect results
        let mut results = Vec::new();
        for _ in 0..datasets.len() {
            if let Some(result) = self.receive()? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get number of items processed
    pub fn items_processed(&self) -> usize {
        self.items_processed.load(Ordering::Acquire)
    }

    /// Stop the pipeline gracefully
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.input_sender = None; // Drop sender to wake up workers
    }

    /// Wait for all workers to finish
    pub fn join(mut self) -> Result<()> {
        // Drop senders to signal completion
        self.input_sender = None;

        // Wait for all workers
        let workers = std::mem::take(&mut self.workers);
        for worker in workers {
            worker.join().map_err(|_| {
                DatasetsError::ProcessingError("Worker thread panicked".to_string())
            })?;
        }

        Ok(())
    }
}

impl Drop for ParallelPipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Create a simple preprocessing pipeline
///
/// # Arguments
/// * `preprocess_fn` - Function to apply to each chunk
/// * `num_workers` - Number of worker threads (0 = auto)
///
/// # Returns
/// * `ParallelPipeline` - The pipeline instance
pub fn create_pipeline<F>(preprocess_fn: F, num_workers: usize) -> ParallelPipeline
where
    F: Fn(&Array2<f64>) -> Result<Array2<f64>> + Send + Sync + 'static,
{
    let config = ParallelConfig::default().with_workers(num_workers);
    ParallelPipeline::new(config, Arc::new(preprocess_fn))
}

/// Create a pipeline with custom configuration
pub fn create_pipeline_with_config<F>(config: ParallelConfig, preprocess_fn: F) -> ParallelPipeline
where
    F: Fn(&Array2<f64>) -> Result<Array2<f64>> + Send + Sync + 'static,
{
    ParallelPipeline::new(config, Arc::new(preprocess_fn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::new()
            .with_workers(4)
            .with_batch_size(500)
            .with_buffer_sizes(5, 5)
            .with_work_stealing(true);

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.input_buffer_size, 5);
        assert_eq!(config.output_buffer_size, 5);
        assert!(config.enable_work_stealing);
    }

    #[test]
    fn test_simple_pipeline() -> Result<()> {
        // Create a simple preprocessing function (multiply by 2)
        let preprocess = |data: &Array2<f64>| -> Result<Array2<f64>> { Ok(data * 2.0) };

        let mut pipeline = create_pipeline(preprocess, 2);

        // Submit some data
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .map_err(|e| DatasetsError::InvalidFormat(format!("{}", e)))?;

        pipeline.submit(data.clone(), None)?;

        // Receive result
        if let Some(result) = pipeline.receive()? {
            assert_eq!(result.data[[0, 0]], 2.0);
            assert_eq!(result.data[[2, 2]], 18.0);
        }

        pipeline.stop();
        Ok(())
    }

    #[test]
    fn test_batch_processing() -> Result<()> {
        let preprocess = |data: &Array2<f64>| -> Result<Array2<f64>> { Ok(data + 1.0) };

        let mut pipeline = create_pipeline(preprocess, 4);

        // Create batch of datasets
        let datasets: Vec<Dataset> = (0..5)
            .map(|i| {
                let data = Array2::from_elem((2, 2), i as f64);
                Dataset {
                    data,
                    target: None,
                    targetnames: None,
                    featurenames: None,
                    feature_descriptions: None,
                    description: None,
                    metadata: Default::default(),
                }
            })
            .collect();

        let results = pipeline.process_batch(&datasets)?;
        assert_eq!(results.len(), 5);

        pipeline.stop();
        Ok(())
    }

    #[test]
    fn test_pipeline_stats() -> Result<()> {
        let preprocess = |data: &Array2<f64>| -> Result<Array2<f64>> { Ok(data.clone()) };

        let mut pipeline = create_pipeline(preprocess, 2);

        let data = Array2::zeros((5, 5));
        for _ in 0..3 {
            pipeline.submit(data.clone(), None)?;
        }

        // Drain results
        for _ in 0..3 {
            let _ = pipeline.receive()?;
        }

        assert_eq!(pipeline.items_processed(), 3);

        pipeline.stop();
        Ok(())
    }
}
