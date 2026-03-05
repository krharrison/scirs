//! Out-of-core processing for datasets larger than available RAM.
//!
//! This module provides [`OutOfCoreProcessor`], a lightweight utility that
//! divides a slice into chunks (optionally overlapping) and applies
//! user-supplied functions to each chunk either sequentially or in parallel.
//! Unlike the file-backed I/O utilities in `crate::memory::out_of_core`, this
//! module operates entirely on in-memory slices and is intended for pipelined,
//! chunk-at-a-time processing where the full dataset is too large to hold in
//! cache at once.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::out_of_core::OutOfCoreProcessor;
//!
//! let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
//! let processor = OutOfCoreProcessor::new(100);
//!
//! // Sum each chunk independently.
//! let chunk_sums: Vec<f64> = processor.map(&data, |chunk| chunk.iter().sum());
//! assert_eq!(chunk_sums.len(), 10);
//!
//! // Compute a global sum via fold.
//! let total: f64 = processor.fold(&data, 0.0, |acc, chunk| {
//!     acc + chunk.iter().sum::<f64>()
//! });
//! let expected: f64 = (0..1000).map(|x| x as f64).sum();
//! assert!((total - expected).abs() < 1e-9);
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// OutOfCoreProcessor
// ──────────────────────────────────────────────────────────────────────────────

/// Chunked processor for large in-memory slices.
///
/// `T` is the element type.  The processor splits the input slice into
/// non-overlapping chunks of `chunk_size` elements (the last chunk may be
/// shorter) and exposes sequential and parallel map/fold operations.
///
/// When `overlap > 0`, each window passed to [`OutOfCoreProcessor::windowed_map`]
/// is padded on both sides by `overlap` elements from the neighbouring data.
#[derive(Debug, Clone)]
pub struct OutOfCoreProcessor<T> {
    chunk_size: usize,
    overlap: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> OutOfCoreProcessor<T> {
    /// Create a new processor with the given chunk size and no overlap.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be greater than zero");
        Self {
            chunk_size,
            overlap: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new processor with both a chunk size and an overlap for
    /// windowed operations.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is zero.
    pub fn with_overlap(chunk_size: usize, overlap: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be greater than zero");
        Self {
            chunk_size,
            overlap,
            _marker: std::marker::PhantomData,
        }
    }

    /// The configured chunk size.
    #[inline]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// The configured overlap.
    #[inline]
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Compute the number of non-overlapping chunks for a slice of `len` elements.
    pub fn num_chunks(&self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (len + self.chunk_size - 1) / self.chunk_size
        }
    }

    // ── Sequential map ────────────────────────────────────────────────────────

    /// Apply `func` to each non-overlapping chunk and collect the results.
    ///
    /// Chunks are processed in order.  The last chunk may be smaller than
    /// `chunk_size` when `data.len()` is not a multiple of `chunk_size`.
    ///
    /// # Returns
    ///
    /// A `Vec<R>` with one element per chunk.
    pub fn map<F, R>(&self, data: &[T], func: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R,
    {
        data.chunks(self.chunk_size).map(func).collect()
    }

    // ── Sequential fold ───────────────────────────────────────────────────────

    /// Fold over non-overlapping chunks, threading an accumulator through each.
    ///
    /// Chunks are visited in order from left to right.
    pub fn fold<F, S>(&self, data: &[T], initial: S, func: F) -> S
    where
        F: Fn(S, &[T]) -> S,
    {
        data.chunks(self.chunk_size).fold(initial, func)
    }

    // ── Windowed map ──────────────────────────────────────────────────────────

    /// Apply `func` to overlapping windows of the data.
    ///
    /// Each window covers `chunk_size + 2 * overlap` elements (or fewer at the
    /// boundaries).  The window is aligned so that the *centre* of the window
    /// corresponds to the same non-overlapping chunk that `map` would produce;
    /// at the boundaries the window is truncated rather than padded.
    ///
    /// # Returns
    ///
    /// A `Vec<R>` with one element per non-overlapping chunk.
    pub fn windowed_map<F, R>(&self, data: &[T], func: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R,
    {
        let n = data.len();
        if n == 0 {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(self.num_chunks(n));
        let mut chunk_start = 0;

        while chunk_start < n {
            let chunk_end = (chunk_start + self.chunk_size).min(n);
            // Extend the window by `overlap` on each side, clamped to [0, n).
            let win_start = chunk_start.saturating_sub(self.overlap);
            let win_end = (chunk_end + self.overlap).min(n);
            results.push(func(&data[win_start..win_end]));
            chunk_start = chunk_end;
        }

        results
    }
}

// Parallel operations require T: Sync and F: Sync.
impl<T: Sync> OutOfCoreProcessor<T> {
    // ── Parallel map ─────────────────────────────────────────────────────────

    /// Apply `func` to each non-overlapping chunk in parallel using Rayon.
    ///
    /// The output order matches the chunk order even though execution may be
    /// unordered.
    ///
    /// When the `parallel` feature is disabled this falls back to the
    /// sequential [`map`](OutOfCoreProcessor::map) implementation.
    pub fn par_map<F, R>(&self, data: &[T], func: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            data.par_chunks(self.chunk_size).map(func).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.map(data, func)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_stores_chunk_size() {
        let p: OutOfCoreProcessor<i32> = OutOfCoreProcessor::new(64);
        assert_eq!(p.chunk_size(), 64);
        assert_eq!(p.overlap(), 0);
    }

    #[test]
    fn test_with_overlap_stores_both() {
        let p: OutOfCoreProcessor<f64> = OutOfCoreProcessor::with_overlap(128, 16);
        assert_eq!(p.chunk_size(), 128);
        assert_eq!(p.overlap(), 16);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than zero")]
    fn test_new_zero_chunk_panics() {
        let _: OutOfCoreProcessor<u8> = OutOfCoreProcessor::new(0);
    }

    // ── num_chunks ────────────────────────────────────────────────────────────

    #[test]
    fn test_num_chunks_exact_multiple() {
        let p: OutOfCoreProcessor<u8> = OutOfCoreProcessor::new(10);
        assert_eq!(p.num_chunks(100), 10);
    }

    #[test]
    fn test_num_chunks_remainder() {
        let p: OutOfCoreProcessor<u8> = OutOfCoreProcessor::new(10);
        assert_eq!(p.num_chunks(105), 11);
    }

    #[test]
    fn test_num_chunks_zero_data() {
        let p: OutOfCoreProcessor<u8> = OutOfCoreProcessor::new(10);
        assert_eq!(p.num_chunks(0), 0);
    }

    // ── map ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_map_produces_correct_chunk_count() {
        let data: Vec<i32> = (0..100).collect();
        let p = OutOfCoreProcessor::new(10);
        let sums: Vec<i32> = p.map(&data, |chunk| chunk.iter().sum());
        assert_eq!(sums.len(), 10);
    }

    #[test]
    fn test_map_last_chunk_shorter() {
        let data: Vec<i32> = (0..25).collect();
        let p = OutOfCoreProcessor::new(10);
        let sizes: Vec<usize> = p.map(&data, |chunk| chunk.len());
        assert_eq!(sizes, vec![10, 10, 5]);
    }

    #[test]
    fn test_map_empty_data() {
        let data: Vec<i32> = Vec::new();
        let p = OutOfCoreProcessor::new(10);
        let result: Vec<i32> = p.map(&data, |chunk| chunk.iter().sum());
        assert!(result.is_empty());
    }

    #[test]
    fn test_map_global_sum() {
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let p = OutOfCoreProcessor::new(100);
        let chunk_sums: Vec<f64> = p.map(&data, |c| c.iter().sum());
        let total: f64 = chunk_sums.iter().sum();
        let expected: f64 = (0..1000).map(|x: i64| x as f64).sum();
        assert!((total - expected).abs() < 1e-9);
    }

    // ── fold ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_fold_global_sum() {
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let p = OutOfCoreProcessor::new(100);
        let total = p.fold(&data, 0.0, |acc, chunk| acc + chunk.iter().sum::<f64>());
        let expected: f64 = (0..1000).map(|x: i64| x as f64).sum();
        assert!((total - expected).abs() < 1e-9);
    }

    #[test]
    fn test_fold_count_elements() {
        let data: Vec<u8> = vec![1; 237];
        let p = OutOfCoreProcessor::new(50);
        let count = p.fold(&data, 0usize, |acc, chunk| acc + chunk.len());
        assert_eq!(count, 237);
    }

    #[test]
    fn test_fold_empty_data() {
        let data: Vec<f64> = Vec::new();
        let p = OutOfCoreProcessor::new(10);
        let result = p.fold(&data, 42.0_f64, |acc, _| acc + 1.0);
        assert!((result - 42.0).abs() < 1e-12);
    }

    // ── par_map ───────────────────────────────────────────────────────────────

    #[test]
    fn test_par_map_matches_sequential() {
        let data: Vec<f64> = (0..500).map(|x| x as f64).collect();
        let p = OutOfCoreProcessor::new(50);
        let seq: Vec<f64> = p.map(&data, |c| c.iter().sum());
        let par: Vec<f64> = p.par_map(&data, |c| c.iter().sum());
        assert_eq!(seq, par);
    }

    #[test]
    fn test_par_map_empty() {
        let data: Vec<i32> = Vec::new();
        let p = OutOfCoreProcessor::new(10);
        let result: Vec<i32> = p.par_map(&data, |c| c.iter().sum());
        assert!(result.is_empty());
    }

    // ── windowed_map ──────────────────────────────────────────────────────────

    #[test]
    fn test_windowed_map_chunk_count_matches_map() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let p = OutOfCoreProcessor::with_overlap(10, 2);
        let windowed: Vec<usize> = p.windowed_map(&data, |w| w.len());
        assert_eq!(windowed.len(), 10);
    }

    #[test]
    fn test_windowed_map_interior_chunks_wider() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let p = OutOfCoreProcessor::with_overlap(10, 2);
        let sizes: Vec<usize> = p.windowed_map(&data, |w| w.len());
        // Interior chunks (not first or last) should be chunk_size + 2*overlap = 14.
        for &sz in &sizes[1..sizes.len() - 1] {
            assert_eq!(sz, 14);
        }
    }

    #[test]
    fn test_windowed_map_zero_overlap_equals_map() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let p_window = OutOfCoreProcessor::with_overlap(10, 0);
        let p_plain = OutOfCoreProcessor::new(10);
        let windowed: Vec<f64> = p_window.windowed_map(&data, |w| w.iter().sum());
        let plain: Vec<f64> = p_plain.map(&data, |w| w.iter().sum());
        assert_eq!(windowed, plain);
    }

    #[test]
    fn test_windowed_map_empty_data() {
        let data: Vec<f64> = Vec::new();
        let p = OutOfCoreProcessor::with_overlap(10, 2);
        let result: Vec<f64> = p.windowed_map(&data, |w| w.iter().sum());
        assert!(result.is_empty());
    }
}
