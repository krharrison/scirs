//! Permutation utilities for combinatorial algorithms.
//!
//! This module provides a `Permutation` struct and associated functions for
//! creating, applying, and analyzing permutations.  All operations are
//! implemented in pure Rust without `unwrap()`.

use std::fmt;

/// Error type for permutation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermutationError {
    /// The slice does not form a valid permutation of {0, 1, …, n-1}.
    InvalidPermutation(String),
    /// A length mismatch was encountered.
    LengthMismatch { expected: usize, got: usize },
}

impl fmt::Display for PermutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPermutation(msg) => write!(f, "invalid permutation: {msg}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for PermutationError {}

/// A permutation of the set {0, 1, …, n-1}.
///
/// The permutation is stored as a vector `perm` where `perm[i]` is the image
/// of element `i`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permutation {
    perm: Vec<usize>,
}

impl Permutation {
    /// Create a new permutation from a vector, validating that it is a proper
    /// permutation of {0, …, n-1}.
    ///
    /// # Errors
    ///
    /// Returns [`PermutationError::InvalidPermutation`] if the vector is not a
    /// valid permutation.
    pub fn new(perm: Vec<usize>) -> Result<Self, PermutationError> {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &v in &perm {
            if v >= n {
                return Err(PermutationError::InvalidPermutation(format!(
                    "element {v} is out of range for permutation of length {n}"
                )));
            }
            if seen[v] {
                return Err(PermutationError::InvalidPermutation(format!(
                    "element {v} appears more than once"
                )));
            }
            seen[v] = true;
        }
        Ok(Self { perm })
    }

    /// Return the length of the permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Return `true` if the permutation has length zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Access the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.perm
    }

    /// Consume the permutation and return the inner vector.
    #[inline]
    pub fn into_vec(self) -> Vec<usize> {
        self.perm
    }

    /// Return the image of element `i` under this permutation.
    ///
    /// Returns `None` if `i >= self.len()`.
    #[inline]
    pub fn image_of(&self, i: usize) -> Option<usize> {
        self.perm.get(i).copied()
    }

    /// Apply this permutation to a slice, producing a new `Vec<T>`.
    ///
    /// The output satisfies `out[self.perm[i]] = slice[i]` for every `i`.
    ///
    /// # Errors
    ///
    /// Returns [`PermutationError::LengthMismatch`] when `slice.len() != self.len()`.
    pub fn apply_to<T: Clone>(&self, slice: &[T]) -> Result<Vec<T>, PermutationError> {
        let n = self.perm.len();
        if slice.len() != n {
            return Err(PermutationError::LengthMismatch {
                expected: n,
                got: slice.len(),
            });
        }
        let mut out = slice.to_vec();
        for (i, &dest) in self.perm.iter().enumerate() {
            out[dest] = slice[i].clone();
        }
        Ok(out)
    }

    /// Apply this permutation to a slice **in-place** using a temporary buffer.
    ///
    /// After the call `slice[self.perm[i]] == original_slice[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`PermutationError::LengthMismatch`] when lengths differ.
    pub fn apply_inplace<T: Clone>(&self, slice: &mut Vec<T>) -> Result<(), PermutationError> {
        let new_vec = self.apply_to(slice.as_slice())?;
        *slice = new_vec;
        Ok(())
    }

    /// Compute the inverse permutation P^{-1} such that
    /// `P.inverse().apply_to(P.apply_to(v)) == v`.
    pub fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0usize; n];
        for (i, &dest) in self.perm.iter().enumerate() {
            inv[dest] = i;
        }
        // Safety: inv is a valid permutation by construction.
        Self { perm: inv }
    }

    /// Compute the parity (sign) of the permutation.
    ///
    /// Returns `1` for even permutations (product of an even number of
    /// transpositions) and `-1` for odd permutations.
    pub fn parity(&self) -> i64 {
        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut parity = 1i64;
        for start in 0..n {
            if visited[start] {
                continue;
            }
            // Traverse the cycle beginning at `start`.
            let mut cycle_len = 0usize;
            let mut cur = start;
            while !visited[cur] {
                visited[cur] = true;
                cur = self.perm[cur];
                cycle_len += 1;
            }
            // A cycle of length k contributes (k-1) transpositions.
            if cycle_len % 2 == 0 {
                parity = -parity;
            }
        }
        parity
    }

    /// Compose `self` with `other`: the result `p` satisfies
    /// `p.perm[i] = other.perm[self.perm[i]]`.
    ///
    /// # Errors
    ///
    /// Returns [`PermutationError::LengthMismatch`] if the lengths differ.
    pub fn compose(&self, other: &Permutation) -> Result<Permutation, PermutationError> {
        let n = self.perm.len();
        if other.perm.len() != n {
            return Err(PermutationError::LengthMismatch {
                expected: n,
                got: other.perm.len(),
            });
        }
        let perm: Vec<usize> = self.perm.iter().map(|&i| other.perm[i]).collect();
        Ok(Permutation { perm })
    }

    /// Advance to the lexicographically next permutation in-place.
    ///
    /// Returns `true` if successful; returns `false` (and leaves the
    /// permutation unchanged) when already at the last permutation.
    pub fn next_permutation(&mut self) -> bool {
        next_permutation_slice(&mut self.perm)
    }

    /// Return the number of inversions in the permutation.
    ///
    /// An inversion is a pair (i, j) with i < j and perm[i] > perm[j].
    /// The number of inversions equals (perm.len() - parity) / 2 only in a
    /// loose sense; this computes the exact count in O(n²).
    pub fn inversion_count(&self) -> usize {
        let n = self.perm.len();
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.perm[i] > self.perm[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Decompose this permutation into disjoint cycles.
    ///
    /// Returns a `Vec` of cycles; each cycle is a `Vec<usize>` starting
    /// from the smallest element of that cycle.  Fixed points (1-cycles)
    /// are included.
    pub fn cycle_decomposition(&self) -> Vec<Vec<usize>> {
        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut cur = start;
            while !visited[cur] {
                visited[cur] = true;
                cycle.push(cur);
                cur = self.perm[cur];
            }
            cycles.push(cycle);
        }
        cycles
    }
}

impl fmt::Display for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.perm.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Return the identity permutation of length `n`.
pub fn identity_permutation(n: usize) -> Permutation {
    Permutation {
        perm: (0..n).collect(),
    }
}

/// Generate a uniformly random permutation of {0, …, n-1} using a
/// Fisher-Yates shuffle driven by the supplied RNG.
///
/// The `rng` parameter must implement `rand::Rng`.
pub fn random_permutation<R: rand::Rng>(n: usize, rng: &mut R) -> Permutation {
    let mut perm: Vec<usize> = (0..n).collect();
    // Fisher-Yates / Knuth shuffle.
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        perm.swap(i, j);
    }
    Permutation { perm }
}

/// Apply a permutation (given as a slice) to a vector, producing a new `Vec<T>`.
///
/// `out[perm[i]] = data[i]` for every `i`.
///
/// # Errors
///
/// Returns [`PermutationError::LengthMismatch`] when lengths differ.
pub fn apply_permutation<T: Clone>(
    data: &[T],
    perm: &[usize],
) -> Result<Vec<T>, PermutationError> {
    let n = perm.len();
    if data.len() != n {
        return Err(PermutationError::LengthMismatch {
            expected: n,
            got: data.len(),
        });
    }
    let mut out = data.to_vec();
    for (i, &dest) in perm.iter().enumerate() {
        out[dest] = data[i].clone();
    }
    Ok(out)
}

/// Compute the inverse of a permutation slice.
///
/// The result `inv` satisfies `inv[perm[i]] = i`.
///
/// # Errors
///
/// Returns [`PermutationError::InvalidPermutation`] if `perm` is not a valid
/// permutation of {0, …, n-1}.
pub fn inverse_permutation(perm: &[usize]) -> Result<Vec<usize>, PermutationError> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    let mut seen = vec![false; n];
    for (i, &v) in perm.iter().enumerate() {
        if v >= n {
            return Err(PermutationError::InvalidPermutation(format!(
                "element {v} is out of range for length {n}"
            )));
        }
        if seen[v] {
            return Err(PermutationError::InvalidPermutation(format!(
                "element {v} appears more than once"
            )));
        }
        seen[v] = true;
        inv[v] = i;
    }
    Ok(inv)
}

/// Compute the parity of a permutation slice.
///
/// Returns `1` for even and `-1` for odd permutations.
///
/// # Errors
///
/// Returns [`PermutationError::InvalidPermutation`] if `perm` is not valid.
pub fn permutation_parity(perm: &[usize]) -> Result<i64, PermutationError> {
    let p = Permutation::new(perm.to_vec())?;
    Ok(p.parity())
}

/// Advance a mutable slice to the next lexicographic permutation.
///
/// Returns `true` when the permutation was advanced; returns `false` when
/// `perm` was already the last (descending) permutation and has been reset
/// to the first (ascending) one — callers can use this as a wraparound
/// signal.
///
/// This is the standard Narayana / next-permutation algorithm (O(n) worst
/// case per step, O(1) average).
pub fn next_permutation_slice(perm: &mut [usize]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    // 1. Find the largest index i such that perm[i] < perm[i+1].
    let mut i = n - 1;
    loop {
        if i == 0 {
            // Permutation is in descending order — no next permutation.
            perm.reverse();
            return false;
        }
        i -= 1;
        if perm[i] < perm[i + 1] {
            break;
        }
    }
    // 2. Find the largest j > i such that perm[i] < perm[j].
    let mut j = n - 1;
    while perm[j] <= perm[i] {
        j -= 1;
    }
    // 3. Swap perm[i] and perm[j].
    perm.swap(i, j);
    // 4. Reverse the suffix starting at i+1.
    perm[i + 1..].reverse();
    true
}

// ---------------------------------------------------------------------------
// PermutationIterator
// ---------------------------------------------------------------------------

/// An iterator that yields all n! permutations of {0, …, n-1} in
/// lexicographic order, starting from the identity permutation.
///
/// The iterator owns its current state and advances using the standard
/// next-permutation algorithm, so it uses O(n) memory.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::permutations::PermutationIterator;
///
/// let perms: Vec<Vec<usize>> = PermutationIterator::new(3)
///     .map(|p| p.into_vec())
///     .collect();
/// assert_eq!(perms.len(), 6);
/// assert_eq!(perms[0], vec![0, 1, 2]);
/// assert_eq!(perms[5], vec![2, 1, 0]);
/// ```
pub struct PermutationIterator {
    current: Vec<usize>,
    first: bool,
    done: bool,
}

impl PermutationIterator {
    /// Create a new iterator over all permutations of {0, …, n-1}.
    pub fn new(n: usize) -> Self {
        Self {
            current: (0..n).collect(),
            first: true,
            done: n == 0,
        }
    }
}

impl Iterator for PermutationIterator {
    type Item = Permutation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.first {
            self.first = false;
            return Some(Permutation {
                perm: self.current.clone(),
            });
        }
        // Advance to the next permutation.
        let advanced = next_permutation_slice(&mut self.current);
        if !advanced {
            // We wrapped around: iteration complete.
            self.done = true;
            return None;
        }
        Some(Permutation {
            perm: self.current.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_permutation() {
        let p = identity_permutation(4);
        assert_eq!(p.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_apply_permutation() {
        // perm = [1, 2, 0]: sends element at position 0 → slot 1, etc.
        let perm = [1usize, 2, 0];
        let data = [10, 20, 30];
        let out = apply_permutation(&data, &perm).expect("apply should succeed");
        // out[1]=data[0]=10, out[2]=data[1]=20, out[0]=data[2]=30
        assert_eq!(out, vec![30, 10, 20]);
    }

    #[test]
    fn test_inverse_permutation() {
        let perm = vec![1usize, 2, 0, 3];
        let inv = inverse_permutation(&perm).expect("inverse should succeed");
        // Apply perm then inv should give identity.
        let data = vec![5usize, 6, 7, 8];
        let out1 = apply_permutation(&data, &perm).expect("apply");
        let out2 = apply_permutation(&out1, &inv).expect("apply inv");
        assert_eq!(out2, data);
    }

    #[test]
    fn test_parity_identity() {
        let parity = permutation_parity(&[0usize, 1, 2, 3]).expect("parity");
        assert_eq!(parity, 1);
    }

    #[test]
    fn test_parity_transposition() {
        // Swap 0 and 1: one transposition → odd.
        let parity = permutation_parity(&[1usize, 0, 2, 3]).expect("parity");
        assert_eq!(parity, -1);
    }

    #[test]
    fn test_parity_three_cycle() {
        // 3-cycle (0 1 2) = (01)(02) → even.
        let parity = permutation_parity(&[1usize, 2, 0]).expect("parity");
        assert_eq!(parity, 1);
    }

    #[test]
    fn test_next_permutation() {
        let mut perm = vec![0usize, 1, 2];
        let advanced = next_permutation_slice(&mut perm);
        assert!(advanced);
        assert_eq!(perm, vec![0, 2, 1]);
    }

    #[test]
    fn test_permutation_iterator_count() {
        let count = PermutationIterator::new(4).count();
        assert_eq!(count, 24); // 4! = 24
    }

    #[test]
    fn test_permutation_iterator_order() {
        let perms: Vec<Vec<usize>> = PermutationIterator::new(3)
            .map(|p| p.into_vec())
            .collect();
        assert_eq!(perms[0], vec![0, 1, 2]);
        assert_eq!(perms[1], vec![0, 2, 1]);
        assert_eq!(perms[2], vec![1, 0, 2]);
        assert_eq!(perms[5], vec![2, 1, 0]);
    }

    #[test]
    fn test_cycle_decomposition() {
        // Permutation (0 1 2)(3) — 3-cycle and fixed point.
        let p = Permutation::new(vec![1, 2, 0, 3]).expect("valid");
        let cycles = p.cycle_decomposition();
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_invalid_permutation() {
        let result = Permutation::new(vec![0, 0, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_permutation() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let p = random_permutation(8, &mut rng);
        assert_eq!(p.len(), 8);
        // Verify it is a valid permutation.
        let mut sorted = p.into_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_compose() {
        let a = Permutation::new(vec![1, 0, 2]).expect("valid");
        let b = Permutation::new(vec![2, 1, 0]).expect("valid");
        let ab = a.compose(&b).expect("compose");
        // a maps 0→1, b maps 1→1: ab maps 0→1
        // a maps 1→0, b maps 0→2: ab maps 1→2
        // a maps 2→2, b maps 2→0: ab maps 2→0
        assert_eq!(ab.as_slice(), &[1, 2, 0]);
    }
}
