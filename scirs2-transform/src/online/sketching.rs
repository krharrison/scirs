//! Probabilistic data structure sketches for streaming data.
//!
//! Provides memory-efficient approximate data structures:
//! - [`CountMinSketch`]: frequency estimation under the turnstile model.
//! - [`CountSketch`]: pairwise-independent frequency sketch.
//! - [`BloomFilter`]: set membership with bounded false-positive probability.
//! - [`HyperLogLog`]: cardinality estimation (distinct count).
//! - [`ReservoirSampler`]: uniform sampling from a stream.

use crate::error::{Result, TransformError};

/// Count-Min Sketch for frequency estimation.
///
/// Cormode & Muthukrishnan (2005): "An Improved Data Stream Summary:
/// The Count-Min Sketch and its Applications".
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<i64>>,
    seeds: Vec<u64>,
}

impl CountMinSketch {
    /// Create a Count-Min Sketch with specified width and depth.
    ///
    /// Width ≥ ⌈e / ε⌉ gives error within ε‖f‖₁ with probability ≥ 1 − δ
    /// when depth ≥ ⌈ln(1/δ)⌉.
    pub fn new(width: usize, depth: usize) -> Self {
        let seeds: Vec<u64> = (0..depth as u64).map(|i| i * 0x517cc1b727220a95 + 1).collect();
        Self {
            width,
            depth,
            table: vec![vec![0i64; width]; depth],
            seeds,
        }
    }

    fn hash(&self, item: u64, row: usize) -> usize {
        let h = item.wrapping_mul(self.seeds[row]).wrapping_add(0x9e3779b97f4a7c15);
        (h as usize) % self.width
    }

    /// Increment the count of `item` by `delta`.
    pub fn update(&mut self, item: u64, delta: i64) {
        for row in 0..self.depth {
            let col = self.hash(item, row);
            self.table[row][col] += delta;
        }
    }

    /// Estimate the frequency of `item`.
    pub fn estimate(&self, item: u64) -> i64 {
        (0..self.depth)
            .map(|row| {
                let col = self.hash(item, row);
                self.table[row][col]
            })
            .min()
            .unwrap_or(0)
    }
}

/// Count Sketch for pairwise-independent frequency estimation.
#[derive(Debug, Clone)]
pub struct CountSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<i64>>,
    h_seeds: Vec<u64>,
    s_seeds: Vec<u64>,
}

impl CountSketch {
    /// Create a Count Sketch.
    pub fn new(width: usize, depth: usize) -> Self {
        let h_seeds: Vec<u64> = (0..depth as u64).map(|i| i * 0xbf58476d1ce4e5b9 + 7).collect();
        let s_seeds: Vec<u64> = (0..depth as u64).map(|i| i * 0x94d049bb133111eb + 3).collect();
        Self {
            width,
            depth,
            table: vec![vec![0i64; width]; depth],
            h_seeds,
            s_seeds,
        }
    }

    fn hash(&self, item: u64, row: usize) -> usize {
        let h = item.wrapping_mul(self.h_seeds[row]).wrapping_add(0x9e3779b97f4a7c15);
        (h as usize) % self.width
    }

    fn sign(&self, item: u64, row: usize) -> i64 {
        let s = item.wrapping_mul(self.s_seeds[row]);
        if s & 1 == 0 { 1 } else { -1 }
    }

    /// Update the sketch with `item` and weight `delta`.
    pub fn update(&mut self, item: u64, delta: i64) {
        for row in 0..self.depth {
            let col = self.hash(item, row);
            let sgn = self.sign(item, row);
            self.table[row][col] += sgn * delta;
        }
    }

    /// Estimate the frequency of `item` (median estimator).
    pub fn estimate(&self, item: u64) -> i64 {
        let mut estimates: Vec<i64> = (0..self.depth)
            .map(|row| {
                let col = self.hash(item, row);
                let sgn = self.sign(item, row);
                sgn * self.table[row][col]
            })
            .collect();
        estimates.sort_unstable();
        estimates[self.depth / 2]
    }
}

/// Bloom Filter for approximate set membership.
///
/// Bloom (1970): "Space/time trade-offs in hash coding with allowable errors".
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<bool>,
    num_bits: usize,
    num_hashes: usize,
    seeds: Vec<u64>,
}

impl BloomFilter {
    /// Create a Bloom Filter for `capacity` elements with false-positive rate `fp_rate`.
    pub fn new(capacity: usize, fp_rate: f64) -> Result<Self> {
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(TransformError::InvalidInput(
                "fp_rate must be in (0, 1)".to_string(),
            ));
        }
        let ln2 = std::f64::consts::LN_2;
        let num_bits = (-(capacity as f64) * fp_rate.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bits = num_bits.max(1);
        let num_hashes = ((num_bits as f64 / capacity as f64) * ln2).ceil() as usize;
        let num_hashes = num_hashes.max(1);
        let seeds: Vec<u64> = (0..num_hashes as u64)
            .map(|i| i.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142))
            .collect();
        Ok(Self {
            bits: vec![false; num_bits],
            num_bits,
            num_hashes,
            seeds,
        })
    }

    fn hash(&self, item: u64, k: usize) -> usize {
        let h = item
            .wrapping_mul(self.seeds[k])
            .wrapping_add(0x9e3779b97f4a7c15);
        (h as usize) % self.num_bits
    }

    /// Insert `item` into the filter.
    pub fn insert(&mut self, item: u64) {
        for k in 0..self.num_hashes {
            let idx = self.hash(item, k);
            self.bits[idx] = true;
        }
    }

    /// Query whether `item` might be in the filter.
    ///
    /// Returns `false` means definitely absent; `true` means possibly present.
    pub fn contains(&self, item: u64) -> bool {
        (0..self.num_hashes).all(|k| self.bits[self.hash(item, k)])
    }

    /// Approximate false-positive rate given `n` inserted elements.
    pub fn fp_rate(&self, n: usize) -> f64 {
        let k = self.num_hashes as f64;
        let m = self.num_bits as f64;
        (1.0 - (-k * n as f64 / m).exp()).powf(k)
    }
}

/// HyperLogLog cardinality estimator.
///
/// Flajolet et al. (2007): "HyperLogLog: the analysis of a near-optimal cardinality
/// estimation algorithm".
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    b: u32,
    m: usize,
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Create a HyperLogLog with `b` bits of precision (4 ≤ b ≤ 18).
    pub fn new(b: u32) -> Result<Self> {
        if b < 4 || b > 18 {
            return Err(TransformError::InvalidInput(
                "b must be in [4, 18]".to_string(),
            ));
        }
        let m = 1usize << b;
        Ok(Self {
            b,
            m,
            registers: vec![0u8; m],
        })
    }

    fn alpha(&self) -> f64 {
        match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / self.m as f64),
        }
    }

    fn hash64(item: u64) -> u64 {
        // MurmurHash3 finalizer
        let mut x = item;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        x
    }

    /// Add an element to the sketch.
    pub fn add(&mut self, item: u64) {
        let h = Self::hash64(item);
        let idx = (h >> (64 - self.b)) as usize;
        let w = h << self.b;
        let rho = w.leading_zeros() as u8 + 1;
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Estimate the number of distinct elements seen.
    pub fn cardinality(&self) -> f64 {
        let m = self.m as f64;
        let z = 1.0
            / self
                .registers
                .iter()
                .map(|&r| 2.0_f64.powi(-(r as i32)))
                .sum::<f64>();
        let mut estimate = self.alpha() * m * m * z;
        // Small range correction
        if estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                estimate = m * (m / zeros).ln();
            }
        }
        // Large range correction
        let two32 = (1u64 << 32) as f64;
        if estimate > two32 / 30.0 {
            estimate = -two32 * (1.0 - estimate / two32).ln();
        }
        estimate
    }
}

/// Reservoir Sampler for uniform sampling from a stream.
///
/// Vitter (1985): "Random sampling with a reservoir".
#[derive(Debug, Clone)]
pub struct ReservoirSampler<T> {
    reservoir: Vec<T>,
    capacity: usize,
    seen: usize,
    rng_state: u64,
}

impl<T: Clone> ReservoirSampler<T> {
    /// Create a reservoir sampler with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            seen: 0,
            rng_state: 0x6c62272e07bb0142,
        }
    }

    fn next_rand(&mut self) -> f64 {
        // xorshift64 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    fn rand_usize(&mut self, n: usize) -> usize {
        let r = self.next_rand();
        (r * n as f64) as usize
    }

    /// Insert an item into the reservoir.
    pub fn insert(&mut self, item: T) {
        self.seen += 1;
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(item);
        } else {
            let j = self.rand_usize(self.seen);
            if j < self.capacity {
                self.reservoir[j] = item;
            }
        }
    }

    /// Return a reference to the current reservoir sample.
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Number of elements seen so far.
    pub fn n_seen(&self) -> usize {
        self.seen
    }
}
