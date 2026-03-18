//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::memory_efficient::chunked::ChunkingStrategy;
use crate::memory_efficient::memmap::MemoryMappedArray;
use crate::memory_efficient::memmap_chunks::MemoryMappedChunks;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::types::{
    AdaptiveChunkingBuilder, AdaptiveChunkingParams, AdaptiveChunkingResult, ChunkProcessingStats,
    DynamicChunkAdjuster, MemoryLimits, MemoryPressureLevel, MemoryPressureMonitor, MemoryTrend,
    WorkloadType,
};

/// Shared memory pressure monitor for multi-threaded processing.
pub type SharedMemoryMonitor = Arc<MemoryPressureMonitor>;
/// Create a shared memory pressure monitor.
pub fn create_shared_monitor(limits: MemoryLimits) -> SharedMemoryMonitor {
    Arc::new(MemoryPressureMonitor::new(limits))
}
/// Trait for adaptive chunking capabilities.
pub trait AdaptiveChunking<A: Clone + Copy + 'static + Send + Sync> {
    /// Calculate an optimal chunking strategy based on array characteristics.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    ///
    /// # Returns
    ///
    /// A result containing the recommended chunking strategy and metadata
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult>;
    /// Process chunks using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R;
    /// Process chunks mutably using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize);
    /// Process chunks in parallel using an automatically determined optimal chunking strategy.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to guide the adaptive chunking process
    /// * `f` - Function to process each chunk
    ///
    /// # Returns
    ///
    /// A vector of results, one for each chunk
    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync;
}
impl<A: Clone + Copy + 'static + Send + Sync> AdaptiveChunking<A> for MemoryMappedArray<A> {
    fn adaptive_chunking(
        &self,
        params: AdaptiveChunkingParams,
    ) -> CoreResult<AdaptiveChunkingResult> {
        let total_elements = self.size;
        let elementsize = std::mem::size_of::<A>();
        if elementsize == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Cannot chunk zero-sized type".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        let mut chunksize = params
            .target_memory_usage
            .checked_div(elementsize)
            .ok_or_else(|| {
                CoreError::ComputationError(
                    ErrorContext::new("Arithmetic overflow in chunk size calculation".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        chunksize = chunksize.clamp(params.min_chunksize, params.max_chunksize);
        chunksize = chunksize.min(total_elements);
        let (chunksize, decision_factors) = self.optimize_for_dimensionality(chunksize, &params)?;
        let (chunksize, decision_factors) = if params.optimize_for_parallel {
            let (parallel_chunksize, parallel_factors) =
                self.optimize_for_parallel_processing(chunksize, decision_factors, &params);
            let (final_chunksize, mut final_factors) =
                self.optimize_for_dimensionality(parallel_chunksize, &params)?;
            final_factors.extend(parallel_factors);
            (final_chunksize, final_factors)
        } else {
            (chunksize, decision_factors)
        };
        let strategy = ChunkingStrategy::Fixed(chunksize);
        let estimated_memory = chunksize.checked_mul(elementsize).ok_or_else(|| {
            CoreError::ComputationError(
                ErrorContext::new("Arithmetic overflow in memory estimation".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(AdaptiveChunkingResult {
            strategy,
            estimated_memory_per_chunk: estimated_memory,
            decision_factors,
        })
    }
    fn process_chunks_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R,
    {
        let adaptive_result = self.adaptive_chunking(params)?;
        Ok(self.process_chunks(adaptive_result.strategy, f))
    }
    fn process_chunks_mut_adaptive<F>(
        &mut self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<()>
    where
        F: Fn(&mut [A], usize),
    {
        let adaptive_result = self.adaptive_chunking(params)?;
        self.process_chunks_mut(adaptive_result.strategy, f);
        Ok(())
    }
    #[cfg(feature = "parallel")]
    fn process_chunks_parallel_adaptive<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A], usize) -> R + Send + Sync,
        R: Send,
        A: Send + Sync,
    {
        let mut parallel_params = params;
        parallel_params.optimize_for_parallel = true;
        if parallel_params.numworkers.is_none() {
            parallel_params.numworkers = Some(rayon::current_num_threads());
        }
        let adaptive_result = self.adaptive_chunking(parallel_params)?;
        use crate::memory_efficient::memmap_chunks::MemoryMappedChunksParallel;
        Ok(self.process_chunks_parallel(adaptive_result.strategy, f))
    }
}
impl<A: Clone + Copy + 'static + Send + Sync> MemoryMappedArray<A> {
    /// Optimize chunking based on array dimensionality.
    fn optimize_for_dimensionality(
        &self,
        initial_chunksize: usize,
        params: &AdaptiveChunkingParams,
    ) -> CoreResult<(usize, Vec<String>)> {
        let mut decision_factors = Vec::new();
        let mut chunksize = initial_chunksize;
        match self.shape.len() {
            1 => {
                decision_factors.push("1D array: Using direct chunking".to_string());
            }
            2 => {
                let row_length = self.shape[1];
                if chunksize >= row_length {
                    if chunksize % row_length != 0 {
                        let newsize = (chunksize / row_length)
                            .checked_mul(row_length)
                            .unwrap_or(chunksize);
                        if newsize >= params.min_chunksize {
                            chunksize = newsize;
                            decision_factors
                                .push(
                                    format!(
                                        "2D array: Adjusted chunk size to {chunksize} (multiple of row length {row_length})"
                                    ),
                                );
                        }
                    }
                } else {
                    if row_length <= params.max_chunksize {
                        chunksize = row_length;
                        decision_factors.push(format!(
                            "2D array: Adjusted chunk size to row length {row_length}"
                        ));
                    } else {
                        decision_factors
                            .push(
                                format!(
                                    "2D array: Row length {row_length} exceeds max chunk size, keeping chunk size {chunksize}"
                                ),
                            );
                    }
                }
            }
            3 => {
                let planesize = self.shape[1].checked_mul(self.shape[2]).unwrap_or_else(|| {
                    decision_factors.push(
                        "3D array: Overflow in plane size calculation, using row alignment"
                            .to_string(),
                    );
                    self.shape[2]
                });
                let row_length = self.shape[2];
                if chunksize >= planesize && chunksize % planesize != 0 {
                    let newsize = (chunksize / planesize)
                        .checked_mul(planesize)
                        .unwrap_or(chunksize);
                    if newsize >= params.min_chunksize {
                        chunksize = newsize;
                        decision_factors
                            .push(
                                format!(
                                    "3D array: Adjusted chunk size to {chunksize} (multiple of plane size {planesize})"
                                ),
                            );
                    }
                } else if chunksize >= row_length && chunksize % row_length != 0 {
                    let newsize = (chunksize / row_length)
                        .checked_mul(row_length)
                        .unwrap_or(chunksize);
                    if newsize >= params.min_chunksize {
                        chunksize = newsize;
                        decision_factors
                            .push(
                                format!(
                                    "3D array: Adjusted chunk size to {chunksize} (multiple of row length {row_length})"
                                ),
                            );
                    }
                }
            }
            n => {
                decision_factors.push(format!("{n}D array: Using default chunking strategy"));
            }
        }
        Ok((chunksize, decision_factors))
    }
    /// Optimize chunking for parallel processing.
    fn optimize_for_parallel_processing(
        &self,
        initial_chunksize: usize,
        mut decision_factors: Vec<String>,
        params: &AdaptiveChunkingParams,
    ) -> (usize, Vec<String>) {
        let mut chunksize = initial_chunksize;
        if let Some(numworkers) = params.numworkers {
            let total_elements = self.size;
            let target_num_chunks = numworkers.checked_mul(2).unwrap_or(numworkers);
            let ideal_chunksize = total_elements
                .checked_div(target_num_chunks)
                .unwrap_or(total_elements);
            if ideal_chunksize >= params.min_chunksize && ideal_chunksize <= params.max_chunksize {
                chunksize = ideal_chunksize;
                decision_factors
                    .push(
                        format!(
                            "Parallel optimization: Adjusted chunk size to {chunksize} for {numworkers} workers"
                        ),
                    );
            } else if ideal_chunksize < params.min_chunksize {
                chunksize = params.min_chunksize;
                let actual_chunks = total_elements / chunksize
                    + if total_elements % chunksize != 0 {
                        1
                    } else {
                        0
                    };
                decision_factors
                    .push(
                        format!(
                            "Parallel optimization: Using minimum chunk size {chunksize}, resulting in {actual_chunks} chunks for {numworkers} workers"
                        ),
                    );
            }
        } else {
            decision_factors
                .push(
                    "Parallel optimization requested but no worker count specified, using default chunking"
                        .to_string(),
                );
        }
        (chunksize, decision_factors)
    }
}
/// Extended trait for adaptive chunking with memory monitoring (v0.2.0).
pub trait AdaptiveChunkingWithMonitoring<A: Clone + Copy + 'static + Send + Sync>:
    AdaptiveChunking<A>
{
    /// Process chunks with memory pressure monitoring and dynamic adjustment.
    ///
    /// This method monitors memory pressure during processing and automatically
    /// adjusts chunk sizes to prevent OOM conditions.
    fn process_chunks_monitored<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<(Vec<R>, ChunkProcessingStats)>
    where
        F: Fn(&[A], usize) -> R;
    /// Create a dynamic chunk adjuster based on the params and array characteristics.
    fn create_adjuster(&self, params: &AdaptiveChunkingParams) -> CoreResult<DynamicChunkAdjuster>;
}
impl<A: Clone + Copy + 'static + Send + Sync> AdaptiveChunkingWithMonitoring<A>
    for MemoryMappedArray<A>
{
    fn process_chunks_monitored<F, R>(
        &self,
        params: AdaptiveChunkingParams,
        f: F,
    ) -> CoreResult<(Vec<R>, ChunkProcessingStats)>
    where
        F: Fn(&[A], usize) -> R,
    {
        let start_time = Instant::now();
        let mut stats = ChunkProcessingStats::default();
        let monitor: Option<SharedMemoryMonitor> = if params.enable_oom_prevention {
            params.memory_monitor.clone().or_else(|| {
                params
                    .memory_limits
                    .as_ref()
                    .map(|limits| create_shared_monitor(limits.clone()))
            })
        } else {
            None
        };
        let initial_result = self.adaptive_chunking(params.clone())?;
        let initial_chunk_size = match initial_result.strategy {
            ChunkingStrategy::Fixed(size) => size,
            _ => params.min_chunksize.max(1024),
        };
        let adjuster = if params.enable_dynamic_adjustment {
            let adj = DynamicChunkAdjuster::new(
                initial_chunk_size,
                params.min_chunksize,
                params.max_chunksize,
            );
            if let Some(ref mon) = monitor {
                Some(adj.with_monitor(mon.clone()))
            } else {
                Some(adj)
            }
        } else {
            None
        };
        let mut results = Vec::new();
        let total_elements = self.size;
        let mut processed = 0;
        while processed < total_elements {
            if let Some(ref mon) = monitor {
                if mon.should_pause() {
                    stats.pressure_events += 1;
                    std::thread::sleep(Duration::from_millis(10));
                    if mon.should_pause() {
                        if let Some(ref adj) = adjuster {
                            adj.emergency_reduce();
                            stats.adjustments_made += 1;
                        }
                    }
                }
            }
            let chunk_size = if let Some(ref adj) = adjuster {
                adj.get_chunk_size()
            } else {
                initial_chunk_size
            };
            let end = (processed + chunk_size).min(total_elements);
            let chunk_start = Instant::now();
            let data = self.as_slice();
            let chunk = &data[processed..end];
            let result = f(chunk, stats.chunks_processed);
            results.push(result);
            let chunk_time = chunk_start.elapsed();
            if let Some(ref adj) = adjuster {
                adj.record_chunk_time(chunk_time);
            }
            stats.chunks_processed += 1;
            processed = end;
        }
        stats.total_time = start_time.elapsed();
        stats.avg_chunk_time = if stats.chunks_processed > 0 {
            stats.total_time / stats.chunks_processed as u32
        } else {
            Duration::ZERO
        };
        if let Some(ref adj) = adjuster {
            stats.adjustments_made += adj.adjustment_count();
            stats.final_chunk_size = adj.get_chunk_size();
        } else {
            stats.final_chunk_size = initial_chunk_size;
        }
        stats.peak_memory_estimate = stats.final_chunk_size * std::mem::size_of::<A>();
        Ok((results, stats))
    }
    fn create_adjuster(&self, params: &AdaptiveChunkingParams) -> CoreResult<DynamicChunkAdjuster> {
        let result = self.adaptive_chunking(params.clone())?;
        let initial_size = match result.strategy {
            ChunkingStrategy::Fixed(size) => size,
            _ => params.min_chunksize.max(1024),
        };
        let mut adjuster =
            DynamicChunkAdjuster::new(initial_size, params.min_chunksize, params.max_chunksize);
        if let Some(duration) = params.target_chunk_duration {
            adjuster = adjuster.with_target_time(duration);
        }
        if let Some(ref monitor) = params.memory_monitor {
            adjuster = adjuster.with_monitor(monitor.clone());
        }
        Ok(adjuster)
    }
}
/// Beta 2: Advanced adaptive chunking algorithms and load balancing
pub mod beta2_enhancements {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    /// Performance metrics collector for adaptive optimization
    #[derive(Debug, Clone, Default)]
    #[allow(dead_code)]
    pub struct ChunkingPerformanceMetrics {
        pub chunk_processing_times: Vec<Duration>,
        pub memory_usage_per_chunk: Vec<usize>,
        pub throughput_mbps: Vec<f64>,
        pub cpu_utilization: Vec<f64>,
    }
    /// Beta 2: Dynamic load balancer for heterogeneous computing environments
    #[allow(dead_code)]
    pub struct DynamicLoadBalancer {
        worker_performance: Vec<f64>,
        current_loads: Arc<Vec<AtomicUsize>>,
        target_efficiency: f64,
    }
    #[allow(dead_code)]
    impl DynamicLoadBalancer {
        /// Create a new load balancer for the specified number of workers
        pub fn new(numworkers: usize) -> Self {
            Self {
                worker_performance: vec![1.0; numworkers],
                current_loads: Arc::new((0..numworkers).map(|_| AtomicUsize::new(0)).collect()),
                target_efficiency: 0.85,
            }
        }
        /// Calculate optimal chunk distribution based on worker performance
        pub fn distribute_work(&self, totalwork: usize) -> Vec<usize> {
            let total_performance: f64 = self.worker_performance.iter().sum();
            let mut distribution = Vec::new();
            let mut remaining_work = totalwork;
            for (i, &performance) in self.worker_performance.iter().enumerate() {
                if i == self.worker_performance.len() - 1 {
                    distribution.push(remaining_work);
                } else {
                    let work_share = (totalwork as f64 * performance / total_performance) as usize;
                    distribution.push(work_share);
                    remaining_work = remaining_work.saturating_sub(work_share);
                }
            }
            distribution
        }
        /// Update worker performance metrics based on observed execution times
        pub fn update_performance(
            &mut self,
            workerid: usize,
            work_amount: usize,
            execution_time: Duration,
        ) {
            if workerid < self.worker_performance.len() {
                let performance = work_amount as f64 / execution_time.as_secs_f64();
                let alpha = 0.1;
                self.worker_performance[workerid] =
                    (1.0 - alpha) * self.worker_performance[workerid] + alpha * performance;
            }
        }
    }
    /// Beta 2: Intelligent chunk size predictor using historical data
    #[allow(dead_code)]
    pub struct ChunkSizePredictor {
        historical_metrics: Vec<ChunkingPerformanceMetrics>,
        workload_characteristics: Vec<(WorkloadType, usize)>,
    }
    #[allow(dead_code)]
    impl ChunkSizePredictor {
        pub fn new() -> Self {
            Self {
                historical_metrics: Vec::new(),
                workload_characteristics: Vec::new(),
            }
        }
        /// Predict optimal chunk size based on workload characteristics and history
        pub fn predict_chunk_size(
            &self,
            workload: WorkloadType,
            memory_available: usize,
            data_size: usize,
        ) -> usize {
            let historical_prediction = self.get_historical_prediction(workload);
            let memory_constrained = (memory_available / 4).max(1024);
            let data_constrained = (data_size / 8).max(1024);
            let base_prediction = historical_prediction.unwrap_or(64 * 1024);
            let memory_weight = 0.4;
            let data_weight = 0.4;
            let historical_weight = 0.2;
            let predicted_size = (memory_weight * memory_constrained as f64
                + data_weight * data_constrained as f64
                + historical_weight * base_prediction as f64)
                as usize;
            predicted_size.clamp(1024, 256 * 1024 * 1024)
        }
        fn get_historical_prediction(&self, workload: WorkloadType) -> Option<usize> {
            self.workload_characteristics
                .iter()
                .rev()
                .find(|(wl, _)| *wl == workload)
                .map(|(_, size)| *size)
        }
        /// Record performance metrics for future predictions
        pub fn record_performance(
            &mut self,
            workload: WorkloadType,
            chunk_size: usize,
            metrics: ChunkingPerformanceMetrics,
        ) {
            self.historical_metrics.push(metrics);
            self.workload_characteristics.push((workload, chunk_size));
            if self.historical_metrics.len() > 100 {
                self.historical_metrics.remove(0);
                self.workload_characteristics.remove(0);
            }
        }
    }
    /// Beta 2: NUMA-aware chunking for large multi-socket systems
    #[allow(dead_code)]
    pub fn numa_aware_chunking(data_size: usize, num_numanodes: usize) -> ChunkingStrategy {
        if num_numanodes <= 1 {
            return ChunkingStrategy::Auto;
        }
        let base_chunk_size = data_size / (num_numanodes * 2);
        let aligned_chunk_size = align_to_cache_line(base_chunk_size);
        ChunkingStrategy::Fixed(aligned_chunk_size)
    }
    /// Align size to cache line boundaries for better performance
    fn align_to_cache_line(size: usize) -> usize {
        const CACHE_LINE_SIZE: usize = 64;
        size.div_ceil(CACHE_LINE_SIZE) * CACHE_LINE_SIZE
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    #[test]
    fn test_adaptive_chunking_1d() {
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_adaptive_1d.bin");
        let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Operation failed");
        for val in &data {
            file.write_all(&val.to_ne_bytes())
                .expect("Operation failed");
        }
        drop(file);
        let mmap =
            MemoryMappedArray::<f64>::path(&file_path, &[100_000]).expect("Operation failed");
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(1024 * 1024)
            .with_min_chunksize(1000)
            .with_max_chunksize(50000)
            .optimize_for_parallel(false)
            .build();
        let result = mmap.adaptive_chunking(params).expect("Operation failed");
        match result.strategy {
            ChunkingStrategy::Fixed(chunksize) => {
                assert_eq!(chunksize, 50000);
            }
            _ => panic!("Expected fixed chunking strategy"),
        }
        assert!(result.estimated_memory_per_chunk > 0);
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("1D array")));
    }
    #[test]
    fn test_adaptive_chunking_2d() {
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_adaptive_2d.bin");
        let rows = 1000;
        let cols = 120;
        let data = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64);
        let mut file = File::create(&file_path).expect("Operation failed");
        for val in data.iter() {
            file.write_all(&val.to_ne_bytes())
                .expect("Operation failed");
        }
        drop(file);
        let mmap =
            MemoryMappedArray::<f64>::path(&file_path, &[rows, cols]).expect("Operation failed");
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(100 * 1024)
            .with_min_chunksize(1000)
            .with_max_chunksize(50000)
            .build();
        let result = mmap.adaptive_chunking(params).expect("Operation failed");
        match result.strategy {
            ChunkingStrategy::Fixed(chunksize) => {
                assert_eq!(
                    chunksize % cols,
                    0,
                    "Chunk size should be a multiple of row length"
                );
            }
            _ => panic!("Expected fixed chunking strategy"),
        }
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("2D array")));
    }
    #[test]
    fn test_adaptive_chunking_parallel() {
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_adaptive_parallel.bin");
        let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Operation failed");
        for val in &data {
            file.write_all(&val.to_ne_bytes())
                .expect("Operation failed");
        }
        drop(file);
        let mmap =
            MemoryMappedArray::<f64>::path(&file_path, &[1_000_000]).expect("Operation failed");
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(10 * 1024 * 1024)
            .optimize_for_parallel(true)
            .with_numworkers(4)
            .build();
        let result = mmap.adaptive_chunking(params).expect("Operation failed");
        match result.strategy {
            ChunkingStrategy::Fixed(chunksize) => {
                assert!(chunksize > 0, "Chunk size should be positive");
            }
            _ => panic!("Expected fixed chunking strategy"),
        }
        assert!(result
            .decision_factors
            .iter()
            .any(|s| s.contains("Parallel optimization")));
    }
    #[test]
    fn test_memory_limits_auto_detect() {
        let limits = MemoryLimits::auto_detect();
        assert!(limits.max_memory_usage > 0, "Max memory should be positive");
        assert!(
            limits.reserved_memory > 0,
            "Reserved memory should be positive"
        );
        assert!(limits.pressure_threshold > 0.0 && limits.pressure_threshold < 1.0);
        assert!(limits.critical_threshold > limits.pressure_threshold);
        assert!(limits.auto_monitor);
    }
    #[test]
    fn test_memory_limits_presets() {
        let conservative = MemoryLimits::conservative();
        let aggressive = MemoryLimits::aggressive();
        assert!(conservative.pressure_threshold < aggressive.pressure_threshold);
        assert!(conservative.critical_threshold < aggressive.critical_threshold);
    }
    #[test]
    fn test_memory_pressure_level_reduction_factors() {
        assert_eq!(MemoryPressureLevel::Normal.reduction_factor(), 1.0);
        assert_eq!(MemoryPressureLevel::Elevated.reduction_factor(), 0.75);
        assert_eq!(MemoryPressureLevel::High.reduction_factor(), 0.5);
        assert_eq!(MemoryPressureLevel::Critical.reduction_factor(), 0.25);
        assert!(MemoryPressureLevel::Normal < MemoryPressureLevel::Elevated);
        assert!(MemoryPressureLevel::Elevated < MemoryPressureLevel::High);
        assert!(MemoryPressureLevel::High < MemoryPressureLevel::Critical);
    }
    #[test]
    fn test_memory_pressure_monitor_creation() {
        let limits = MemoryLimits::auto_detect();
        let monitor = MemoryPressureMonitor::new(limits.clone());
        assert_eq!(monitor.get_current_level(), MemoryPressureLevel::Normal);
        assert_eq!(monitor.limits().max_memory_usage, limits.max_memory_usage);
    }
    #[test]
    fn test_memory_pressure_monitor_check() {
        let monitor = MemoryPressureMonitor::with_defaults();
        let level = monitor.check_pressure();
        assert!(matches!(
            level,
            MemoryPressureLevel::Normal
                | MemoryPressureLevel::Elevated
                | MemoryPressureLevel::High
                | MemoryPressureLevel::Critical
        ));
    }
    #[test]
    fn test_memory_pressure_monitor_recommended_size() {
        let monitor = MemoryPressureMonitor::with_defaults();
        let base_size = 100_000;
        let recommended = monitor.recommended_chunk_size(base_size);
        assert!(recommended >= 1024);
        assert!(recommended <= base_size);
    }
    #[test]
    fn test_shared_monitor_creation() {
        let limits = MemoryLimits::auto_detect();
        let monitor = create_shared_monitor(limits);
        let monitor2 = monitor.clone();
        assert_eq!(monitor.get_current_level(), monitor2.get_current_level());
    }
    #[test]
    fn test_dynamic_chunk_adjuster_creation() {
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000);
        assert_eq!(adjuster.get_chunk_size(), 10_000);
        assert_eq!(adjuster.adjustment_count(), 0);
    }
    #[test]
    fn test_dynamic_chunk_adjuster_with_monitor() {
        let monitor = create_shared_monitor(MemoryLimits::auto_detect());
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000).with_monitor(monitor);
        let size = adjuster.get_chunk_size();
        assert!(size >= 1_000);
        assert!(size <= 100_000);
    }
    #[test]
    fn test_dynamic_chunk_adjuster_record_time() {
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000)
            .with_target_time(Duration::from_millis(100));
        for _ in 0..5 {
            adjuster.record_chunk_time(Duration::from_millis(50));
        }
        let _size = adjuster.get_chunk_size();
    }
    #[test]
    fn test_dynamic_chunk_adjuster_emergency_reduce() {
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000);
        adjuster.emergency_reduce();
        assert!(adjuster.get_chunk_size() < 10_000);
        assert_eq!(adjuster.adjustment_count(), 1);
    }
    #[test]
    fn test_dynamic_chunk_adjuster_reset() {
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000);
        adjuster.emergency_reduce();
        assert!(adjuster.get_chunk_size() < 10_000);
        adjuster.reset();
        assert_eq!(adjuster.get_chunk_size(), 10_000);
    }
    #[test]
    fn test_dynamic_chunk_adjuster_disabled() {
        let adjuster = DynamicChunkAdjuster::new(10_000, 1_000, 100_000);
        adjuster.set_enabled(false);
        adjuster.emergency_reduce();
        assert_eq!(adjuster.get_chunk_size(), 10_000);
    }
    #[test]
    fn test_builder_with_memory_limits() {
        let limits = MemoryLimits::conservative();
        let params = AdaptiveChunkingBuilder::new()
            .with_memory_limits(limits.clone())
            .build();
        assert!(params.memory_limits.is_some());
        assert_eq!(
            params.memory_limits.as_ref().map(|l| l.pressure_threshold),
            Some(limits.pressure_threshold)
        );
    }
    #[test]
    fn test_builder_with_oom_prevention() {
        let params = AdaptiveChunkingBuilder::new()
            .with_oom_prevention(true)
            .build();
        assert!(params.enable_oom_prevention);
    }
    #[test]
    fn test_builder_with_dynamic_adjustment() {
        let params = AdaptiveChunkingBuilder::new()
            .with_dynamic_adjustment(true)
            .build();
        assert!(params.enable_dynamic_adjustment);
    }
    #[test]
    fn test_builder_memory_constrained() {
        let default_params = AdaptiveChunkingBuilder::new().build();
        let constrained_params = AdaptiveChunkingBuilder::new().memory_constrained().build();
        assert!(
            constrained_params.target_memory_usage < default_params.target_memory_usage,
            "Memory constrained should have lower target memory"
        );
        assert!(constrained_params.enable_oom_prevention);
        assert!(constrained_params.enable_dynamic_adjustment);
    }
    #[test]
    fn test_builder_high_performance() {
        let default_params = AdaptiveChunkingBuilder::new().build();
        let hp_params = AdaptiveChunkingBuilder::new().high_performance().build();
        assert!(
            hp_params.target_memory_usage >= default_params.target_memory_usage,
            "High performance should have higher or equal target memory"
        );
    }
    #[test]
    fn test_builder_for_workload() {
        let mem_intensive = AdaptiveChunkingBuilder::new()
            .for_workload(WorkloadType::MemoryIntensive)
            .build();
        let compute_intensive = AdaptiveChunkingBuilder::new()
            .for_workload(WorkloadType::ComputeIntensive)
            .build();
        assert!(mem_intensive.memory_limits.is_some());
        assert!(compute_intensive.memory_limits.is_some());
        let mem_threshold = mem_intensive
            .memory_limits
            .as_ref()
            .map(|l| l.pressure_threshold)
            .unwrap_or(1.0);
        let compute_threshold = compute_intensive
            .memory_limits
            .as_ref()
            .map(|l| l.pressure_threshold)
            .unwrap_or(0.0);
        assert!(
            mem_threshold < compute_threshold,
            "Memory intensive should have lower pressure threshold"
        );
    }
    #[test]
    fn test_process_chunks_monitored() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("test_monitored.bin");
        let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Failed to create file");
        for val in &data {
            file.write_all(&val.to_ne_bytes()).expect("Failed to write");
        }
        drop(file);
        let mmap =
            MemoryMappedArray::<f64>::path(&file_path, &[10_000]).expect("Failed to create mmap");
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(1024 * 1024)
            .with_min_chunksize(100)
            .with_max_chunksize(5000)
            .with_oom_prevention(true)
            .with_dynamic_adjustment(true)
            .build();
        let (results, stats) = mmap
            .process_chunks_monitored(params, |chunk, _idx| chunk.iter().sum::<f64>())
            .expect("Processing failed");
        assert!(!results.is_empty(), "Should have results");
        assert!(stats.chunks_processed > 0, "Should have processed chunks");
        assert!(
            stats.total_time.as_nanos() > 0,
            "Should have taken some time"
        );
        assert!(stats.final_chunk_size > 0, "Should have a final chunk size");
    }
    #[test]
    fn test_create_adjuster_from_array() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("test_adjuster.bin");
        let data: Vec<f64> = (0..5_000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Failed to create file");
        for val in &data {
            file.write_all(&val.to_ne_bytes()).expect("Failed to write");
        }
        drop(file);
        let mmap =
            MemoryMappedArray::<f64>::path(&file_path, &[5_000]).expect("Failed to create mmap");
        let params = AdaptiveChunkingBuilder::new()
            .with_min_chunksize(100)
            .with_max_chunksize(2000)
            .with_target_duration(Duration::from_millis(50))
            .build();
        let adjuster = mmap
            .create_adjuster(&params)
            .expect("Failed to create adjuster");
        assert!(adjuster.get_chunk_size() >= 100);
        assert!(adjuster.get_chunk_size() <= 2000);
    }
    #[test]
    fn test_memory_trend_initial() {
        let monitor = MemoryPressureMonitor::with_defaults();
        assert_eq!(monitor.get_trend(), MemoryTrend::Stable);
    }
    #[test]
    fn test_chunk_processing_stats_default() {
        let stats = ChunkProcessingStats::default();
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.adjustments_made, 0);
        assert_eq!(stats.pressure_events, 0);
        assert_eq!(stats.total_time, Duration::ZERO);
    }
}
