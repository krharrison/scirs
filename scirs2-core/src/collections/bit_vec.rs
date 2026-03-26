//! `BitVec` — a compact bit vector backed by a `Vec<u64>`.
//!
//! Each bit is stored in a single machine-word-sized chunk, making storage
//! roughly 64× more memory-efficient than a `Vec<bool>`.  All bitwise
//! operations (AND, OR, XOR) operate word-at-a-time for maximum throughput.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::collections::BitVec;
//!
//! let mut bv = BitVec::new(8);
//! bv.set(2, true);
//! bv.set(5, true);
//!
//! assert!(bv.get(2));
//! assert!(!bv.get(3));
//! assert_eq!(bv.count_ones(), 2);
//!
//! let ones: Vec<usize> = bv.iter_ones().collect();
//! assert_eq!(ones, vec![2, 5]);
//! ```

use std::fmt;

// ============================================================================
// BitVec
// ============================================================================

/// A compact bit vector.
///
/// Bits are stored in `u64` chunks.  The bit at logical index `i` lives in
/// chunk `i / 64` at bit position `i % 64`.
#[derive(Clone)]
pub struct BitVec {
    /// Number of valid logical bits.
    n_bits: usize,
    /// Backing storage; `chunks.len() == ceil(n_bits / 64)`.
    chunks: Vec<u64>,
}

impl BitVec {
    /// Creates a new `BitVec` of `n_bits` bits, all initialised to `false`.
    pub fn new(n_bits: usize) -> Self {
        let n_chunks = Self::chunks_for(n_bits);
        BitVec {
            n_bits,
            chunks: vec![0u64; n_chunks],
        }
    }

    /// Creates a `BitVec` of `n_bits` bits, all initialised to `true`.
    pub fn all_ones(n_bits: usize) -> Self {
        let n_chunks = Self::chunks_for(n_bits);
        let mut chunks = vec![u64::MAX; n_chunks];
        // Zero out the padding bits in the last chunk to keep invariants.
        Self::mask_last_chunk(&mut chunks, n_bits);
        BitVec { n_bits, chunks }
    }

    /// Returns the total number of logical bits.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Returns `true` if the bit vector has zero capacity.
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }

    /// Sets bit `i` to `val`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n_bits()`.
    pub fn set(&mut self, i: usize, val: bool) {
        assert!(
            i < self.n_bits,
            "BitVec::set: index {} out of range (len={})",
            i,
            self.n_bits
        );
        let (chunk_idx, bit_idx) = Self::locate(i);
        if val {
            self.chunks[chunk_idx] |= 1u64 << bit_idx;
        } else {
            self.chunks[chunk_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Returns the value of bit `i`.
    ///
    /// Returns `false` for out-of-range indices (does not panic).
    pub fn get(&self, i: usize) -> bool {
        if i >= self.n_bits {
            return false;
        }
        let (chunk_idx, bit_idx) = Self::locate(i);
        (self.chunks[chunk_idx] >> bit_idx) & 1 == 1
    }

    /// Toggles bit `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n_bits()`.
    pub fn flip(&mut self, i: usize) {
        assert!(i < self.n_bits, "BitVec::flip: index {} out of range", i);
        let (chunk_idx, bit_idx) = Self::locate(i);
        self.chunks[chunk_idx] ^= 1u64 << bit_idx;
    }

    /// Returns the number of bits set to `1`.
    pub fn count_ones(&self) -> usize {
        self.chunks.iter().map(|c| c.count_ones() as usize).sum()
    }

    /// Returns the number of bits set to `0`.
    pub fn count_zeros(&self) -> usize {
        self.n_bits - self.count_ones()
    }

    /// Returns an iterator over the indices of all `1`-bits in ascending order.
    pub fn iter_ones(&self) -> IterOnes<'_> {
        IterOnes {
            bv: self,
            chunk_idx: 0,
            // Start with the first chunk's bits; will advance as needed.
            remaining: if self.chunks.is_empty() {
                0u64
            } else {
                self.chunks[0]
            },
            logical_base: 0,
        }
    }

    /// Returns an iterator over the indices of all `0`-bits in ascending order.
    pub fn iter_zeros(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n_bits).filter(|&i| !self.get(i))
    }

    /// Clears all bits (sets every bit to `0`).
    pub fn clear(&mut self) {
        for c in &mut self.chunks {
            *c = 0;
        }
    }

    /// Sets all bits to `1`.
    pub fn set_all(&mut self) {
        for c in &mut self.chunks {
            *c = u64::MAX;
        }
        Self::mask_last_chunk(&mut self.chunks, self.n_bits);
    }

    // ------------------------------------------------------------------
    // Bitwise operations (in-place)
    // ------------------------------------------------------------------

    /// `self &= other`.  Panics if the bit-widths differ.
    pub fn and_assign(&mut self, other: &BitVec) {
        self.assert_same_len(other, "and_assign");
        for (a, b) in self.chunks.iter_mut().zip(other.chunks.iter()) {
            *a &= *b;
        }
    }

    /// `self |= other`.  Panics if the bit-widths differ.
    pub fn or_assign(&mut self, other: &BitVec) {
        self.assert_same_len(other, "or_assign");
        for (a, b) in self.chunks.iter_mut().zip(other.chunks.iter()) {
            *a |= *b;
        }
    }

    /// `self ^= other`.  Panics if the bit-widths differ.
    pub fn xor_assign(&mut self, other: &BitVec) {
        self.assert_same_len(other, "xor_assign");
        for (a, b) in self.chunks.iter_mut().zip(other.chunks.iter()) {
            *a ^= *b;
        }
    }

    // ------------------------------------------------------------------
    // Bitwise operations (returning new BitVec)
    // ------------------------------------------------------------------

    /// Returns `self & other`.  Panics if the bit-widths differ.
    pub fn and(&self, other: &BitVec) -> BitVec {
        self.assert_same_len(other, "and");
        let chunks: Vec<u64> = self
            .chunks
            .iter()
            .zip(other.chunks.iter())
            .map(|(a, b)| a & b)
            .collect();
        BitVec {
            n_bits: self.n_bits,
            chunks,
        }
    }

    /// Returns `self | other`.  Panics if the bit-widths differ.
    pub fn or(&self, other: &BitVec) -> BitVec {
        self.assert_same_len(other, "or");
        let chunks: Vec<u64> = self
            .chunks
            .iter()
            .zip(other.chunks.iter())
            .map(|(a, b)| a | b)
            .collect();
        BitVec {
            n_bits: self.n_bits,
            chunks,
        }
    }

    /// Returns `self ^ other`.  Panics if the bit-widths differ.
    pub fn xor(&self, other: &BitVec) -> BitVec {
        self.assert_same_len(other, "xor");
        let chunks: Vec<u64> = self
            .chunks
            .iter()
            .zip(other.chunks.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        BitVec {
            n_bits: self.n_bits,
            chunks,
        }
    }

    /// Returns the bitwise NOT of `self`.
    pub fn not(&self) -> BitVec {
        let mut chunks: Vec<u64> = self.chunks.iter().map(|c| !c).collect();
        Self::mask_last_chunk(&mut chunks, self.n_bits);
        BitVec {
            n_bits: self.n_bits,
            chunks,
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    #[inline]
    fn chunks_for(n_bits: usize) -> usize {
        (n_bits + 63) / 64
    }

    #[inline]
    fn locate(i: usize) -> (usize, usize) {
        (i / 64, i % 64)
    }

    /// Zeros out bits beyond `n_bits` in the last chunk.
    fn mask_last_chunk(chunks: &mut Vec<u64>, n_bits: usize) {
        if n_bits == 0 || chunks.is_empty() {
            return;
        }
        let tail = n_bits % 64;
        if tail != 0 {
            let last = chunks.len() - 1;
            chunks[last] &= (1u64 << tail) - 1;
        }
    }

    fn assert_same_len(&self, other: &BitVec, op: &str) {
        assert_eq!(
            self.n_bits, other.n_bits,
            "BitVec::{}: length mismatch ({} vs {})",
            op, self.n_bits, other.n_bits
        );
    }
}

// ============================================================================
// IterOnes
// ============================================================================

/// Iterator over the indices of set bits, produced by [`BitVec::iter_ones`].
pub struct IterOnes<'a> {
    bv: &'a BitVec,
    /// Index of the *current* chunk we are scanning.
    chunk_idx: usize,
    /// The bits still to be processed in the current chunk.
    remaining: u64,
    /// Logical bit index of bit 0 in the current chunk.
    logical_base: usize,
}

impl<'a> Iterator for IterOnes<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.remaining != 0 {
                // Fast-path: consume the lowest set bit.
                let bit_pos = self.remaining.trailing_zeros() as usize;
                // Clear that bit.
                self.remaining &= self.remaining - 1;
                let logical = self.logical_base + bit_pos;
                if logical < self.bv.n_bits {
                    return Some(logical);
                }
                // Bit is a padding bit — skip.
                continue;
            }
            // Move to next chunk.
            self.chunk_idx += 1;
            if self.chunk_idx >= self.bv.chunks.len() {
                return None;
            }
            self.logical_base = self.chunk_idx * 64;
            self.remaining = self.bv.chunks[self.chunk_idx];
        }
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl PartialEq for BitVec {
    fn eq(&self, other: &Self) -> bool {
        self.n_bits == other.n_bits && self.chunks == other.chunks
    }
}

impl Eq for BitVec {}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec({} bits: ", self.n_bits)?;
        for i in 0..self.n_bits {
            write!(f, "{}", if self.get(i) { '1' } else { '0' })?;
        }
        write!(f, ")")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get() {
        let mut bv = BitVec::new(64);
        bv.set(0, true);
        bv.set(63, true);
        assert!(bv.get(0));
        assert!(bv.get(63));
        assert!(!bv.get(32));
    }

    #[test]
    fn test_count_ones() {
        let mut bv = BitVec::new(100);
        bv.set(0, true);
        bv.set(50, true);
        bv.set(99, true);
        assert_eq!(bv.count_ones(), 3);
        assert_eq!(bv.count_zeros(), 97);
    }

    #[test]
    fn test_iter_ones() {
        let mut bv = BitVec::new(200);
        let expected = vec![0usize, 63, 64, 127, 199];
        for &i in &expected {
            bv.set(i, true);
        }
        let ones: Vec<_> = bv.iter_ones().collect();
        assert_eq!(ones, expected);
    }

    #[test]
    fn test_bitwise_and() {
        let mut a = BitVec::new(8);
        let mut b = BitVec::new(8);
        a.set(1, true);
        a.set(3, true);
        b.set(1, true);
        b.set(5, true);
        let c = a.and(&b);
        assert!(c.get(1));
        assert!(!c.get(3));
        assert!(!c.get(5));
    }

    #[test]
    fn test_bitwise_or() {
        let mut a = BitVec::new(8);
        let mut b = BitVec::new(8);
        a.set(1, true);
        b.set(5, true);
        let c = a.or(&b);
        assert!(c.get(1));
        assert!(c.get(5));
    }

    #[test]
    fn test_bitwise_xor() {
        let mut a = BitVec::new(8);
        let mut b = BitVec::new(8);
        a.set(1, true);
        a.set(3, true);
        b.set(1, true);
        b.set(5, true);
        let c = a.xor(&b);
        assert!(!c.get(1), "1 XOR 1 = 0");
        assert!(c.get(3), "1 XOR 0 = 1");
        assert!(c.get(5), "0 XOR 1 = 1");
    }

    #[test]
    fn test_not() {
        let mut bv = BitVec::new(8);
        bv.set(0, true);
        bv.set(7, true);
        let n = bv.not();
        assert!(!n.get(0));
        assert!(!n.get(7));
        assert!(n.get(1));
        assert_eq!(n.count_ones(), 6);
    }

    #[test]
    fn test_all_ones() {
        let bv = BitVec::all_ones(10);
        assert_eq!(bv.count_ones(), 10);
        for i in 0..10 {
            assert!(bv.get(i));
        }
    }

    #[test]
    fn test_flip() {
        let mut bv = BitVec::new(16);
        bv.flip(4);
        assert!(bv.get(4));
        bv.flip(4);
        assert!(!bv.get(4));
    }

    #[test]
    fn test_large_bitvec() {
        let n = 10_000;
        let mut bv = BitVec::new(n);
        for i in (0..n).step_by(7) {
            bv.set(i, true);
        }
        let expected_count = (0..n).step_by(7).count();
        assert_eq!(bv.count_ones(), expected_count);

        let ones: Vec<_> = bv.iter_ones().collect();
        let expected_ones: Vec<_> = (0..n).step_by(7).collect();
        assert_eq!(ones, expected_ones);
    }

    #[test]
    fn test_clear_and_set_all() {
        let mut bv = BitVec::new(128);
        bv.set_all();
        assert_eq!(bv.count_ones(), 128);
        bv.clear();
        assert_eq!(bv.count_ones(), 0);
    }

    #[test]
    fn test_out_of_range_get() {
        let bv = BitVec::new(8);
        assert!(!bv.get(100)); // should not panic, just return false
    }

    #[test]
    fn test_clone_eq() {
        let mut bv = BitVec::new(32);
        bv.set(5, true);
        bv.set(15, true);
        let bv2 = bv.clone();
        assert_eq!(bv, bv2);
    }
}
