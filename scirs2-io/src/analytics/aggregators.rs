//! Streaming aggregation functions with incremental updates.
//!
//! Provides numerically stable aggregators using Kahan compensated summation
//! and Welford's online algorithm for mean/variance computation.

use std::collections::HashMap;

/// Trait for streaming aggregators.
///
/// All aggregators are `Send` so they can be used across thread boundaries.
pub trait StreamAggregator: Send {
    /// The type of values ingested by this aggregator.
    type Input: Clone + Send;
    /// The type of the aggregated result.
    type Output: Clone + Send;

    /// Incorporate a new value into the running state.
    fn update(&mut self, value: Self::Input);
    /// Return the current aggregated result without consuming the state.
    fn result(&self) -> Self::Output;
    /// Merge another aggregator's state into `self` (useful for parallel reduce).
    fn merge(&mut self, other: &Self);
    /// Reset state to the initial (empty) condition.
    fn reset(&mut self);
    /// Return the number of values seen since the last reset.
    fn count(&self) -> usize;
}

/// Count aggregator — counts the number of events seen.
#[derive(Debug, Clone)]
pub struct CountAgg {
    count: usize,
}

impl CountAgg {
    /// Create a new, empty `CountAgg`.
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl Default for CountAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAggregator for CountAgg {
    type Input = f64;
    type Output = usize;

    fn update(&mut self, _: f64) {
        self.count += 1;
    }

    fn result(&self) -> usize {
        self.count
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Sum aggregator using Kahan compensated summation for numerical stability.
#[derive(Debug, Clone)]
pub struct SumAgg {
    sum: f64,
    compensation: f64,
    count: usize,
}

impl SumAgg {
    /// Create a new, empty `SumAgg`.
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
            count: 0,
        }
    }
}

impl Default for SumAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAggregator for SumAgg {
    type Input = f64;
    type Output = f64;

    fn update(&mut self, value: f64) {
        // Kahan compensated summation
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
        self.count += 1;
    }

    fn result(&self) -> f64 {
        self.sum
    }

    fn merge(&mut self, other: &Self) {
        self.update(other.sum);
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
        self.count = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Welford online mean/variance aggregator — numerically stable, single-pass.
#[derive(Debug, Clone)]
pub struct MeanAgg {
    count: usize,
    mean: f64,
    m2: f64, // aggregate of squared deviations (for variance)
}

impl MeanAgg {
    /// Create a new, empty `MeanAgg`.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Sample variance (Bessel-corrected).
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Sample standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for MeanAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAggregator for MeanAgg {
    type Input = f64;
    type Output = f64;

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn result(&self) -> f64 {
        self.mean
    }

    fn merge(&mut self, other: &Self) {
        // Parallel Welford merge (Chan et al.)
        let total = self.count + other.count;
        if total == 0 {
            return;
        }
        let delta = other.mean - self.mean;
        self.m2 = self.m2
            + other.m2
            + delta * delta * self.count as f64 * other.count as f64 / total as f64;
        self.mean = (self.mean * self.count as f64 + other.mean * other.count as f64)
            / total as f64;
        self.count = total;
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Min/Max streaming aggregator.
#[derive(Debug, Clone)]
pub struct MinMaxAgg {
    min: f64,
    max: f64,
    count: usize,
}

impl MinMaxAgg {
    /// Create a new, empty `MinMaxAgg`.
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            count: 0,
        }
    }

    /// Range between max and min.
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Current minimum value.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Current maximum value.
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for MinMaxAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAggregator for MinMaxAgg {
    type Input = f64;
    type Output = (f64, f64);

    fn update(&mut self, value: f64) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.count += 1;
    }

    fn result(&self) -> (f64, f64) {
        (self.min, self.max)
    }

    fn merge(&mut self, other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.count += other.count;
    }

    fn reset(&mut self) {
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.count = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Streaming histogram with fixed-width bins, underflow/overflow tracking,
/// percentile queries, and Shannon entropy computation.
#[derive(Debug, Clone)]
pub struct HistogramAgg {
    bins: Vec<u64>,
    min: f64,
    max: f64,
    bin_width: f64,
    n_bins: usize,
    count: usize,
    overflow: u64,
    underflow: u64,
}

impl HistogramAgg {
    /// Create a histogram covering `[min, max)` divided into `n_bins` equal-width bins.
    pub fn new(min: f64, max: f64, n_bins: usize) -> Self {
        assert!(n_bins > 0, "n_bins must be positive");
        assert!(max > min, "max must be greater than min");
        let bin_width = (max - min) / n_bins as f64;
        Self {
            bins: vec![0; n_bins],
            min,
            max,
            bin_width,
            n_bins,
            count: 0,
            overflow: 0,
            underflow: 0,
        }
    }

    /// Returns the left edge of each bin, plus the right edge of the last bin.
    pub fn bin_edges(&self) -> Vec<f64> {
        (0..=self.n_bins)
            .map(|i| self.min + i as f64 * self.bin_width)
            .collect()
    }

    /// Approximate percentile via bin midpoint interpolation.
    pub fn percentile(&self, p: f64) -> f64 {
        let target = (p / 100.0 * self.count as f64) as u64;
        let mut cumsum = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumsum += count;
            if cumsum >= target {
                return self.min + (i as f64 + 0.5) * self.bin_width;
            }
        }
        self.max
    }

    /// Shannon entropy of the empirical distribution.
    pub fn entropy(&self) -> f64 {
        let n = self.count as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.bins
            .iter()
            .map(|&c| {
                if c > 0 {
                    let p = c as f64 / n;
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Number of values that fell below `min`.
    pub fn overflow(&self) -> u64 {
        self.overflow
    }

    /// Number of values that fell at or above `max`.
    pub fn underflow(&self) -> u64 {
        self.underflow
    }
}

impl StreamAggregator for HistogramAgg {
    type Input = f64;
    type Output = Vec<u64>;

    fn update(&mut self, value: f64) {
        self.count += 1;
        if value < self.min {
            self.underflow += 1;
            return;
        }
        if value >= self.max {
            self.overflow += 1;
            return;
        }
        let bin = ((value - self.min) / self.bin_width) as usize;
        let bin = bin.min(self.n_bins - 1);
        self.bins[bin] += 1;
    }

    fn result(&self) -> Vec<u64> {
        self.bins.clone()
    }

    fn merge(&mut self, other: &Self) {
        for (a, b) in self.bins.iter_mut().zip(other.bins.iter()) {
            *a += b;
        }
        self.count += other.count;
        self.overflow += other.overflow;
        self.underflow += other.underflow;
    }

    fn reset(&mut self) {
        self.bins = vec![0; self.n_bins];
        self.count = 0;
        self.overflow = 0;
        self.underflow = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Top-K tracker maintaining the highest-scoring keyed items.
///
/// Uses a sorted Vec (suitable for small K). For very large K consider a
/// `BinaryHeap`-based implementation.
#[derive(Debug, Clone)]
pub struct TopKAgg {
    k: usize,
    heap: Vec<(f64, String)>, // (score, key)
    count: usize,
    /// Secondary index for O(k) dedup during update.
    key_index: HashMap<String, usize>,
}

impl TopKAgg {
    /// Create a new `TopKAgg` that retains the top `k` scored keys.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: Vec::new(),
            count: 0,
            key_index: HashMap::new(),
        }
    }

    /// Update score for a key, maintaining the top-K invariant.
    ///
    /// If the key already exists in the top-K, its score is updated to the
    /// maximum of old and new. Otherwise it is inserted and the lowest-scoring
    /// entry is evicted if the set exceeds K.
    pub fn update_keyed(&mut self, key: String, score: f64) {
        self.count += 1;
        if let Some(&pos) = self.key_index.get(&key) {
            if pos < self.heap.len() && self.heap[pos].1 == key {
                if score > self.heap[pos].0 {
                    self.heap[pos].0 = score;
                }
            } else {
                self.heap.push((score, key.clone()));
            }
        } else {
            self.heap.push((score, key.clone()));
        }

        // Re-sort descending by score
        self.heap
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        self.heap.truncate(self.k);

        // Rebuild index
        self.key_index.clear();
        for (i, (_, k)) in self.heap.iter().enumerate() {
            self.key_index.insert(k.clone(), i);
        }
    }

    /// Return a slice of `(score, key)` pairs in descending order.
    pub fn top_k(&self) -> &[(f64, String)] {
        &self.heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_agg() {
        let mut agg = CountAgg::new();
        for v in [1.0, 2.0, 3.0] {
            agg.update(v);
        }
        assert_eq!(agg.result(), 3);
        assert_eq!(agg.count(), 3);

        let mut other = CountAgg::new();
        other.update(4.0);
        agg.merge(&other);
        assert_eq!(agg.result(), 4);

        agg.reset();
        assert_eq!(agg.result(), 0);
    }

    #[test]
    fn test_sum_kahan_compensation() {
        let mut agg = SumAgg::new();
        // Values that would lose precision with naive summation
        let values = vec![1.0, 1e-10, -1.0, 1e-10];
        for v in &values {
            agg.update(*v);
        }
        let expected: f64 = values.iter().sum();
        assert!(
            (agg.result() - expected).abs() < 1e-15,
            "Kahan sum failed: {} vs {}",
            agg.result(),
            expected
        );
    }

    #[test]
    fn test_mean_welford_parallel_merge() {
        let values_a: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let values_b: Vec<f64> = (6..=10).map(|i| i as f64).collect();

        let mut agg_a = MeanAgg::new();
        let mut agg_b = MeanAgg::new();

        for v in &values_a {
            agg_a.update(*v);
        }
        for v in &values_b {
            agg_b.update(*v);
        }

        agg_a.merge(&agg_b);

        // Mean of 1..=10 = 5.5
        assert!(
            (agg_a.result() - 5.5).abs() < 1e-10,
            "Parallel merge mean: {}",
            agg_a.result()
        );
        // Variance of 1..=10 (sample) = 55/9 ≈ 9.1667
        let expected_var = 55.0 / 9.0;
        assert!(
            (agg_a.variance() - expected_var).abs() < 1e-10,
            "Parallel merge variance: {}",
            agg_a.variance()
        );
    }

    #[test]
    fn test_minmax_agg() {
        let mut agg = MinMaxAgg::new();
        for v in [3.0, 1.0, 4.0, 1.0, 5.0] {
            agg.update(v);
        }
        let (min, max) = agg.result();
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(agg.range(), 4.0);
    }

    #[test]
    fn test_histogram_agg() {
        let mut agg = HistogramAgg::new(0.0, 10.0, 10);
        for v in [0.5, 1.5, 2.5, 9.5, -1.0, 10.5] {
            agg.update(v);
        }
        assert_eq!(agg.count(), 6);
        assert_eq!(agg.underflow(), 1);
        assert_eq!(agg.overflow(), 1);
        let bins = agg.result();
        assert_eq!(bins[0], 1); // 0.5 in [0,1)
        assert_eq!(bins[1], 1); // 1.5 in [1,2)
        assert_eq!(bins[9], 1); // 9.5 in [9,10)
    }

    #[test]
    fn test_topk_agg() {
        let mut agg = TopKAgg::new(3);
        agg.update_keyed("a".to_string(), 1.0);
        agg.update_keyed("b".to_string(), 3.0);
        agg.update_keyed("c".to_string(), 2.0);
        agg.update_keyed("d".to_string(), 0.5);
        let top = agg.top_k();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].1, "b");
    }
}
