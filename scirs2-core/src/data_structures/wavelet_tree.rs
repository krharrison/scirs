//! Wavelet Tree — a compact data structure for sequences over a bounded integer
//! alphabet.
//!
//! A wavelet tree over a sequence `S[0..n)` of symbols drawn from alphabet
//! `[0, σ)` supports:
//!
//! | Operation      | Time complexity | Description                      |
//! |----------------|-----------------|----------------------------------|
//! | `access(i)`    | O(log σ)        | Return the symbol at position i  |
//! | `rank(c, i)`   | O(log σ)        | Count occurrences of c in S[0,i) |
//! | `select(c, k)` | O(log σ)        | Position of k-th occurrence of c |
//!
//! The tree is built by recursively partitioning the alphabet at each level.
//! At the root, symbols `< σ/2` are mapped to bit `0` and symbols `≥ σ/2` to
//! bit `1`.  Left and right child sub-sequences are processed recursively.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::data_structures::WaveletTree;
//!
//! let seq = [3u32, 1, 4, 1, 5, 9, 2, 6];
//! let wt = WaveletTree::build(&seq, 10);
//!
//! assert_eq!(wt.access(0), 3);
//! assert_eq!(wt.access(3), 1);
//! // rank: how many 1s appear in positions [0, 4)?
//! assert_eq!(wt.rank(1, 4), 2);
//! // select: where is the 1st occurrence of symbol 9?
//! assert_eq!(wt.select(9, 1), Some(5));
//! ```

use std::fmt;

// ============================================================================
// BitRankVector — succinct bit vector with O(1) rank queries
// ============================================================================

/// A bit vector augmented with a pre-computed prefix-popcount table for O(1)
/// rank queries.
///
/// `rank1(i)` returns the number of set bits in positions `[0, i)`.
#[derive(Clone, Debug)]
struct BitRankVector {
    /// Raw bit storage (u64 words).
    words: Vec<u64>,
    /// `prefix[k]` = total number of 1-bits in words `[0, k)`.
    prefix: Vec<usize>,
    /// Total number of logical bits.
    len: usize,
}

impl BitRankVector {
    /// Construct from a boolean slice.
    fn from_bits(bits: &[bool]) -> Self {
        let len = bits.len();
        let n_words = (len + 63) / 64;
        let mut words = vec![0u64; n_words];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        // Build prefix table.
        let mut prefix = vec![0usize; n_words + 1];
        for (i, &w) in words.iter().enumerate() {
            prefix[i + 1] = prefix[i] + w.count_ones() as usize;
        }
        BitRankVector { words, prefix, len }
    }

    /// Returns the bit at position `i`.
    #[inline]
    fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Returns the number of 1-bits in `[0, i)` (rank of 1 at position i).
    #[inline]
    fn rank1(&self, i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let i = i.min(self.len);
        let word_idx = (i - 1) / 64;
        let bit_pos = (i - 1) % 64;
        // Popcount up to and including bit `i-1`.
        let mask = if bit_pos == 63 {
            u64::MAX
        } else {
            (1u64 << (bit_pos + 1)) - 1
        };
        self.prefix[word_idx] + (self.words[word_idx] & mask).count_ones() as usize
    }

    /// Returns the number of 0-bits in `[0, i)` (rank of 0 at position i).
    #[inline]
    fn rank0(&self, i: usize) -> usize {
        i.min(self.len) - self.rank1(i)
    }

    /// Total number of logical bits.
    fn len(&self) -> usize {
        self.len
    }
}

// ============================================================================
// WaveletNode — internal node in the wavelet tree
// ============================================================================

/// One level of the wavelet tree.
///
/// Each node covers an alphabet sub-range `[lo, hi)` and stores a bit-vector
/// indicating, for each position in the current sequence, whether its symbol
/// belongs to the upper half `[mid, hi)` (bit = 1) or lower half `[lo, mid)`
/// (bit = 0).
#[derive(Clone, Debug)]
struct WaveletNode {
    /// Alphabet sub-range covered by this node.
    lo: u32,
    hi: u32,
    /// Per-position bits: 0 → symbol goes left, 1 → symbol goes right.
    bits: BitRankVector,
    /// Left child (alphabet `[lo, mid)`).
    left: Option<Box<WaveletNode>>,
    /// Right child (alphabet `[mid, hi)`).
    right: Option<Box<WaveletNode>>,
}

impl WaveletNode {
    /// Recursively build the wavelet tree for `seq` over alphabet `[lo, hi)`.
    fn build(seq: &[u32], lo: u32, hi: u32) -> Option<Box<Self>> {
        if seq.is_empty() || lo + 1 >= hi {
            return None;
        }
        let mid = lo + (hi - lo) / 2;

        // Compute bits: 0 if symbol < mid, 1 if symbol >= mid.
        let bits_raw: Vec<bool> = seq.iter().map(|&s| s >= mid).collect();
        let bits = BitRankVector::from_bits(&bits_raw);

        // Partition sequence into left (symbol < mid) and right (symbol >= mid).
        let left_seq: Vec<u32> = seq.iter().copied().filter(|&s| s < mid).collect();
        let right_seq: Vec<u32> = seq.iter().copied().filter(|&s| s >= mid).collect();

        let left = WaveletNode::build(&left_seq, lo, mid);
        let right = WaveletNode::build(&right_seq, mid, hi);

        Some(Box::new(WaveletNode {
            lo,
            hi,
            bits,
            left,
            right,
        }))
    }

    /// Recursively retrieve the symbol at position `i` (0-indexed).
    fn access(&self, i: usize) -> u32 {
        let mid = self.lo + (self.hi - self.lo) / 2;
        if self.lo + 1 == self.hi {
            return self.lo;
        }
        if !self.bits.get(i) {
            // Symbol is in left child.
            let j = self.bits.rank0(i + 1) - 1;
            match &self.left {
                Some(child) => child.access(j),
                None => self.lo,
            }
        } else {
            // Symbol is in right child.
            let j = self.bits.rank1(i + 1) - 1;
            match &self.right {
                Some(child) => child.access(j),
                None => mid,
            }
        }
    }

    /// Count occurrences of `symbol` in the prefix of length `i` (positions
    /// `[0, i)`).
    fn rank(&self, symbol: u32, i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let mid = self.lo + (self.hi - self.lo) / 2;
        if self.lo + 1 == self.hi {
            // Leaf node: every element equals self.lo.
            return i.min(self.bits.len());
        }
        if symbol < mid {
            // Symbol is in the left half; count 0-bits up to i.
            let j = self.bits.rank0(i);
            match &self.left {
                Some(child) => child.rank(symbol, j),
                None => 0,
            }
        } else {
            // Symbol is in the right half; count 1-bits up to i.
            let j = self.bits.rank1(i);
            match &self.right {
                Some(child) => child.rank(symbol, j),
                None => 0,
            }
        }
    }

    /// Find the position of the `k`-th occurrence (1-indexed) of `symbol`.
    fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        if k == 0 {
            return None;
        }
        let mid = self.lo + (self.hi - self.lo) / 2;
        if self.lo + 1 == self.hi {
            // Leaf: every element here is `self.lo`.
            if k <= self.bits.len() {
                // `select_0` on the all-0 bit vector: position of k-th 0-bit.
                return Some(self.select_zero(k));
            }
            return None;
        }

        if symbol < mid {
            // Recurse into left child to find the k-th occurrence.
            let j = match &self.left {
                Some(child) => child.select(symbol, k)?,
                None => return None,
            };
            // Map the position j (in the left sub-sequence) back to the
            // original sequence by finding the (j+1)-th 0-bit in self.bits.
            Some(self.select_zero(j + 1))
        } else {
            let j = match &self.right {
                Some(child) => child.select(symbol, k)?,
                None => return None,
            };
            // Map j back to original using (j+1)-th 1-bit.
            Some(self.select_one(j + 1))
        }
    }

    /// Position of the `k`-th 0-bit (1-indexed) in `self.bits`.
    fn select_zero(&self, k: usize) -> usize {
        // Linear scan — O(n/64) with word-level acceleration.
        let mut remaining = k;
        for (word_idx, &word) in self.bits.words.iter().enumerate() {
            let zeros = ((!word) as u64).count_ones() as usize;
            let valid_bits = (self.bits.len - word_idx * 64).min(64);
            // Count only zeros in valid bit positions.
            let valid_mask = if valid_bits == 64 {
                u64::MAX
            } else {
                (1u64 << valid_bits) - 1
            };
            let valid_zeros = ((!word) & valid_mask).count_ones() as usize;
            if remaining <= valid_zeros {
                // Answer is in this word.
                let mut w = (!word) & valid_mask;
                for bit_pos in 0..valid_bits {
                    if (w & 1) == 1 {
                        remaining -= 1;
                        if remaining == 0 {
                            return word_idx * 64 + bit_pos;
                        }
                    }
                    w >>= 1;
                }
            }
            remaining -= valid_zeros;
            let _ = zeros;
        }
        self.bits.len() // should not reach here if k is valid
    }

    /// Position of the `k`-th 1-bit (1-indexed) in `self.bits`.
    fn select_one(&self, k: usize) -> usize {
        let mut remaining = k;
        for (word_idx, &word) in self.bits.words.iter().enumerate() {
            let valid_bits = (self.bits.len - word_idx * 64).min(64);
            let valid_mask = if valid_bits == 64 {
                u64::MAX
            } else {
                (1u64 << valid_bits) - 1
            };
            let valid_ones = (word & valid_mask).count_ones() as usize;
            if remaining <= valid_ones {
                let mut w = word & valid_mask;
                for bit_pos in 0..valid_bits {
                    if (w & 1) == 1 {
                        remaining -= 1;
                        if remaining == 0 {
                            return word_idx * 64 + bit_pos;
                        }
                    }
                    w >>= 1;
                }
            }
            remaining -= valid_ones;
        }
        self.bits.len() // should not reach here if k is valid
    }
}

// ============================================================================
// WaveletTree — public API
// ============================================================================

/// A wavelet tree over a sequence of unsigned integers.
///
/// Supports `access`, `rank`, and `select` queries each in O(log σ) time.
pub struct WaveletTree {
    /// The original sequence length.
    n: usize,
    /// The alphabet size (symbols drawn from `[0, sigma)`).
    sigma: u32,
    /// Root of the internal wavelet tree structure.
    root: Option<Box<WaveletNode>>,
}

impl WaveletTree {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Builds a wavelet tree over `seq` with alphabet size `sigma`.
    ///
    /// All values in `seq` must be strictly less than `sigma`.  Values
    /// `>= sigma` are silently treated as `sigma - 1`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::data_structures::WaveletTree;
    /// let wt = WaveletTree::build(&[0u32, 1, 2, 3], 4);
    /// assert_eq!(wt.access(2), 2);
    /// ```
    pub fn build(seq: &[u32], sigma: u32) -> Self {
        let sigma = sigma.max(2);
        let clamped: Vec<u32> = seq.iter().map(|&s| s.min(sigma - 1)).collect();
        let root = WaveletNode::build(&clamped, 0, sigma);
        WaveletTree {
            n: seq.len(),
            sigma,
            root,
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Returns the symbol at position `i` (0-indexed).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn access(&self, i: usize) -> u32 {
        assert!(i < self.n, "index {i} out of bounds (len={})", self.n);
        match &self.root {
            Some(node) => node.access(i),
            None => 0,
        }
    }

    /// Returns the number of occurrences of `symbol` in `seq[0..i)`.
    ///
    /// Returns 0 if `symbol >= sigma` or `i == 0`.
    pub fn rank(&self, symbol: u32, i: usize) -> usize {
        if symbol >= self.sigma || i == 0 {
            return 0;
        }
        let i = i.min(self.n);
        match &self.root {
            Some(node) => node.rank(symbol, i),
            None => 0,
        }
    }

    /// Returns the position (0-indexed) of the `k`-th occurrence (1-indexed)
    /// of `symbol` in the sequence, or `None` if fewer than `k` occurrences
    /// exist.
    pub fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        if symbol >= self.sigma || k == 0 {
            return None;
        }
        match &self.root {
            Some(node) => node.select(symbol, k),
            None => None,
        }
    }

    /// Returns the length of the sequence.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the alphabet size.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }
}

impl fmt::Debug for WaveletTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WaveletTree")
            .field("n", &self.n)
            .field("sigma", &self.sigma)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_seq() -> Vec<u32> {
        vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    }

    #[test]
    fn access_roundtrip() {
        let seq = sample_seq();
        let sigma = 10;
        let wt = WaveletTree::build(&seq, sigma);
        for (i, &expected) in seq.iter().enumerate() {
            assert_eq!(
                wt.access(i),
                expected,
                "access({i}) mismatch"
            );
        }
    }

    #[test]
    fn rank_basic() {
        let seq = sample_seq(); // [3,1,4,1,5,9,2,6,5,3]
        let wt = WaveletTree::build(&seq, 10);

        // rank(1, 5) = number of 1s in seq[0..5] = [3,1,4,1,5] → 2
        assert_eq!(wt.rank(1, 5), 2);
        // rank(3, 10) = full count of 3s = 2
        assert_eq!(wt.rank(3, 10), 2);
        // rank(9, 10) = 1
        assert_eq!(wt.rank(9, 10), 1);
        // rank(7, 10) = 0 (7 not present)
        assert_eq!(wt.rank(7, 10), 0);
        // rank of any symbol at position 0 = 0
        assert_eq!(wt.rank(3, 0), 0);
    }

    #[test]
    fn rank_out_of_sigma() {
        let wt = WaveletTree::build(&[0u32, 1, 2], 3);
        assert_eq!(wt.rank(3, 3), 0);
        assert_eq!(wt.rank(100, 3), 0);
    }

    #[test]
    fn select_basic() {
        let seq = sample_seq(); // [3,1,4,1,5,9,2,6,5,3]
        let wt = WaveletTree::build(&seq, 10);

        // 1st occurrence of 1 → position 1
        assert_eq!(wt.select(1, 1), Some(1));
        // 2nd occurrence of 1 → position 3
        assert_eq!(wt.select(1, 2), Some(3));
        // 3rd occurrence of 1 → None (only two 1s)
        assert_eq!(wt.select(1, 3), None);
        // Only occurrence of 9 → position 5
        assert_eq!(wt.select(9, 1), Some(5));
        // 2nd occurrence of 5 → position 8
        assert_eq!(wt.select(5, 2), Some(8));
    }

    #[test]
    fn select_k_zero_returns_none() {
        let wt = WaveletTree::build(&[1u32, 2, 3], 4);
        assert_eq!(wt.select(1, 0), None);
    }

    #[test]
    fn select_absent_symbol() {
        let wt = WaveletTree::build(&[1u32, 2, 3], 4);
        assert_eq!(wt.select(0, 1), None);
    }

    #[test]
    fn empty_sequence() {
        let wt = WaveletTree::build(&[], 8);
        assert!(wt.is_empty());
        assert_eq!(wt.rank(0, 0), 0);
        assert_eq!(wt.select(0, 1), None);
    }

    #[test]
    fn single_symbol_repeated() {
        let seq = vec![7u32; 20];
        let wt = WaveletTree::build(&seq, 16);
        for i in 0..20 {
            assert_eq!(wt.access(i), 7);
        }
        assert_eq!(wt.rank(7, 20), 20);
        assert_eq!(wt.rank(7, 10), 10);
        for k in 1..=20 {
            assert_eq!(wt.select(7, k), Some(k - 1));
        }
        assert_eq!(wt.select(7, 21), None);
    }

    #[test]
    fn rank_select_consistency() {
        // rank(c, select(c, k)) should equal k for all valid (c, k).
        let seq: Vec<u32> = (0u32..8).flat_map(|c| vec![c, c]).collect();
        let wt = WaveletTree::build(&seq, 8);
        for c in 0u32..8 {
            for k in 1..=2 {
                if let Some(pos) = wt.select(c, k) {
                    let r = wt.rank(c, pos + 1);
                    assert_eq!(r, k, "rank({c}, select({c}, {k})+1) = {r} ≠ {k}");
                }
            }
        }
    }

    #[test]
    fn bit_rank_vector_correctness() {
        let bits = vec![true, false, true, true, false, false, true, false];
        let brv = BitRankVector::from_bits(&bits);
        // rank1(i) = number of 1s in [0, i)
        assert_eq!(brv.rank1(0), 0);
        assert_eq!(brv.rank1(1), 1); // bit 0 = true
        assert_eq!(brv.rank1(2), 1); // bit 1 = false
        assert_eq!(brv.rank1(3), 2); // bit 2 = true
        assert_eq!(brv.rank1(4), 3); // bit 3 = true
        assert_eq!(brv.rank1(8), 4); // 4 ones total
        // rank0
        assert_eq!(brv.rank0(8), 4);
    }
}
