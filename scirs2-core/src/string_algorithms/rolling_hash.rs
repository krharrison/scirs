//! Rolling hash (Rabin-Karp), double rolling hash, edit distance, LCS,
//! and utility hashing functions.
//!
//! The rolling hash supports O(1) window advancement, enabling efficient
//! substring fingerprinting for pattern matching and genomic k-mer analysis.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// RollingHash
// ---------------------------------------------------------------------------

/// Polynomial rolling hash (Rabin-Karp style).
///
/// Computes `H = Σ text[i] * base^(window_len-1-i)  mod modulus` over a
/// sliding window of fixed length.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::RollingHash;
///
/// let mut rh = RollingHash::default_params(3);
/// rh.init(b"abc");
/// let h_abc = rh.hash();
///
/// rh.roll(b'a', b'd'); // window is now "bcd"
/// assert_ne!(rh.hash(), h_abc);
/// ```
#[derive(Debug, Clone)]
pub struct RollingHash {
    base: u64,
    modulus: u64,
    hash: u64,
    window_len: usize,
    /// `powers[i]` = `base^i mod modulus`
    powers: Vec<u64>,
}

impl RollingHash {
    /// Create a rolling hash with explicit base and modulus.
    ///
    /// # Panics
    ///
    /// This function does not panic; invalid `window_len = 0` results in a
    /// zero-length window that always returns hash `0`.
    pub fn new(base: u64, modulus: u64, window_len: usize) -> Self {
        let mut powers = Vec::with_capacity(window_len + 1);
        let mut p = 1u64;
        powers.push(p);
        for _ in 0..window_len {
            p = p.wrapping_mul(base) % modulus;
            powers.push(p);
        }
        RollingHash {
            base,
            modulus,
            hash: 0,
            window_len,
            powers,
        }
    }

    /// Create a rolling hash with well-tested default parameters.
    ///
    /// Uses base = 131, modulus = 2^61 − 1 (a Mersenne prime) truncated to
    /// u64 arithmetic — suitable for general-purpose string hashing.
    pub fn default_params(window_len: usize) -> Self {
        // 2^61 - 1 is a Mersenne prime; we use it modulo u64 with a large prime.
        const BASE: u64 = 131;
        const MOD: u64 = 1_000_000_007;
        Self::new(BASE, MOD, window_len)
    }

    /// Initialize the hash using the first `window_len` bytes of `data`.
    ///
    /// If `data` is shorter than `window_len`, only the available bytes are
    /// hashed (the window is effectively truncated).
    pub fn init(&mut self, data: &[u8]) {
        self.hash = 0;
        let len = data.len().min(self.window_len);
        for &b in &data[..len] {
            self.hash = (self.hash.wrapping_mul(self.base) + b as u64) % self.modulus;
        }
    }

    /// Advance the window by one byte.
    ///
    /// `remove` is the byte leaving the left edge; `add` is the byte entering
    /// the right edge.
    #[inline]
    pub fn roll(&mut self, remove: u8, add: u8) {
        // H_new = (H - remove * base^(len-1)) * base + add
        let leading_power = if self.window_len > 0 { self.powers[self.window_len - 1] } else { 1 };
        let remove_contribution = (remove as u64).wrapping_mul(leading_power) % self.modulus;
        let h = (self.hash + self.modulus - remove_contribution) % self.modulus;
        self.hash = (h.wrapping_mul(self.base) + add as u64) % self.modulus;
    }

    /// Current hash value.
    #[inline]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Window length this hash was built for.
    #[inline]
    pub fn window_len(&self) -> usize {
        self.window_len
    }
}

// ---------------------------------------------------------------------------
// DoubleRollingHash
// ---------------------------------------------------------------------------

/// Double rolling hash — uses two independent polynomial hashes simultaneously.
///
/// Combining two independent hash functions dramatically reduces the
/// probability of false positives in Rabin-Karp search.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::DoubleRollingHash;
///
/// let mut drh = DoubleRollingHash::new(4);
/// drh.init(b"abcd");
/// let (h1, h2) = drh.hash();
/// drh.roll(b'a', b'e');
/// assert_ne!(drh.hash(), (h1, h2));
/// ```
#[derive(Debug, Clone)]
pub struct DoubleRollingHash {
    hash1: RollingHash,
    hash2: RollingHash,
}

impl DoubleRollingHash {
    /// Create a double rolling hash for windows of `window_len` bytes.
    pub fn new(window_len: usize) -> Self {
        DoubleRollingHash {
            hash1: RollingHash::new(131, 1_000_000_007, window_len),
            hash2: RollingHash::new(137, 998_244_353, window_len),
        }
    }

    /// Initialize both hashes from the first `window_len` bytes of `data`.
    pub fn init(&mut self, data: &[u8]) {
        self.hash1.init(data);
        self.hash2.init(data);
    }

    /// Advance both windows by one byte.
    #[inline]
    pub fn roll(&mut self, remove: u8, add: u8) {
        self.hash1.roll(remove, add);
        self.hash2.roll(remove, add);
    }

    /// Return the combined `(hash1, hash2)` pair.
    #[inline]
    pub fn hash(&self) -> (u64, u64) {
        (self.hash1.hash(), self.hash2.hash())
    }
}

// ---------------------------------------------------------------------------
// Rabin-Karp pattern search
// ---------------------------------------------------------------------------

/// Rabin-Karp single-pattern search using a rolling hash.
///
/// Returns all (possibly overlapping) starting positions of `pattern` in
/// `text`.  Candidate positions identified by hash equality are verified
/// with an exact byte comparison to eliminate false positives.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::rabin_karp_search;
///
/// let pos = rabin_karp_search(b"ababab", b"ab");
/// assert_eq!(pos, vec![0, 2, 4]);
/// ```
pub fn rabin_karp_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let n = text.len();
    let m = pattern.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    let mut rh_pat = RollingHash::default_params(m);
    rh_pat.init(pattern);
    let pat_hash = rh_pat.hash();

    let mut rh_txt = RollingHash::default_params(m);
    rh_txt.init(&text[..m]);

    let mut results = Vec::new();

    if rh_txt.hash() == pat_hash && text[..m] == *pattern {
        results.push(0);
    }

    for i in 1..=(n - m) {
        rh_txt.roll(text[i - 1], text[i + m - 1]);
        if rh_txt.hash() == pat_hash && text[i..i + m] == *pattern {
            results.push(i);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Unique substrings via rolling hash
// ---------------------------------------------------------------------------

/// Collect every unique substring of length `k` from `text`.
///
/// Uses a rolling hash to enumerate windows efficiently and a `HashSet` of
/// owned byte vectors for deduplication.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::find_unique_substrings;
///
/// let unique = find_unique_substrings(b"abab", 2);
/// assert_eq!(unique.len(), 2); // "ab" and "ba"
/// ```
pub fn find_unique_substrings(text: &[u8], k: usize) -> HashSet<Vec<u8>> {
    let n = text.len();
    let mut result = HashSet::new();
    if k == 0 || k > n {
        return result;
    }

    for i in 0..=(n - k) {
        result.insert(text[i..i + k].to_vec());
    }
    result
}

// ---------------------------------------------------------------------------
// Count distinct k-mers
// ---------------------------------------------------------------------------

/// Count the number of distinct k-mers (substrings of length `k`) in `text`
/// using a rolling hash for O(n) fingerprinting.
///
/// Candidate duplicates detected by hash collision are verified exactly,
/// so the count is always correct.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::count_distinct_kmers;
///
/// // "abab" has k-mers of length 2: "ab", "ba", "ab" → 2 distinct
/// assert_eq!(count_distinct_kmers(b"abab", 2), 2);
/// ```
pub fn count_distinct_kmers(text: &[u8], k: usize) -> usize {
    let n = text.len();
    if k == 0 || k > n {
        return 0;
    }

    // Map from (hash1, hash2) → list of starting positions with that hash.
    // We keep positions to allow exact verification on collision.
    let mut seen: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    let mut rh = DoubleRollingHash::new(k);
    rh.init(&text[..k]);
    let h = rh.hash();
    let entry = seen.entry(h).or_default();
    // Verify this is genuinely new (first entry, always new).
    entry.push(0);

    let mut distinct = 1usize;

    for i in 1..=(n - k) {
        rh.roll(text[i - 1], text[i + k - 1]);
        let h = rh.hash();
        let entry = seen.entry(h).or_default();
        // Check exact equality against all previously seen positions with
        // the same hash.
        let is_new = entry
            .iter()
            .all(|&prev| text[prev..prev + k] != text[i..i + k]);
        if is_new {
            distinct += 1;
        }
        entry.push(i);
    }

    distinct
}

// ---------------------------------------------------------------------------
// Edit distance (Levenshtein)
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance between byte slices `a` and `b`.
///
/// Uses a full O(|a| × |b|) DP table.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
/// ```
pub fn levenshtein_distance(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Two-row rolling DP.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// Longest Common Subsequence
// ---------------------------------------------------------------------------

/// Compute the length of the longest common subsequence (LCS) of `a` and `b`.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::lcs_length;
///
/// assert_eq!(lcs_length(b"ABCBDAB", b"BDCAB"), 4);
/// ```
pub fn lcs_length(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        for x in curr.iter_mut() {
            *x = 0;
        }
    }

    prev[n]
}

/// Reconstruct one actual LCS sequence from `a` and `b`.
///
/// Builds the full O(mn) DP table and then back-traces to recover the
/// sequence.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::lcs_sequence;
///
/// let seq = lcs_sequence(b"ABCBDAB", b"BDCAB");
/// assert_eq!(seq.len(), 4);
/// ```
pub fn lcs_sequence(a: &[u8], b: &[u8]) -> Vec<u8> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }

    // Full table for back-tracing.
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Back-trace.
    let mut result = Vec::with_capacity(dp[m][n]);
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            result.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result
}

// ---------------------------------------------------------------------------
// String hashing utilities
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a byte slice.
///
/// Fast, non-cryptographic hash suitable for hash-map keys.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::fnv1a_hash;
///
/// let h = fnv1a_hash(b"hello");
/// assert_ne!(h, 0);
/// ```
pub fn fnv1a_hash(s: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &b in s {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// djb2 64-bit hash of a byte slice.
///
/// Classic algorithm attributed to Dan Bernstein.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::djb2_hash;
///
/// let h = djb2_hash(b"hello");
/// assert_ne!(h, 0);
/// ```
pub fn djb2_hash(s: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &b in s {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RollingHash --------------------------------------------------------

    #[test]
    fn test_rolling_hash_init_consistent() {
        let mut rh = RollingHash::default_params(3);
        rh.init(b"abc");
        let h1 = rh.hash();

        let mut rh2 = RollingHash::default_params(3);
        rh2.init(b"abc");
        assert_eq!(rh2.hash(), h1);
    }

    #[test]
    fn test_rolling_hash_roll_correct() {
        // After rolling "abc" → remove 'a', add 'd' => "bcd"
        let mut rh = RollingHash::default_params(3);
        rh.init(b"abcd");
        let h_abc = rh.hash(); // init only uses first 3 bytes

        let mut rh2 = RollingHash::default_params(3);
        rh2.init(b"abc");
        rh2.roll(b'a', b'd');

        // The rolled hash of "abc"→"bcd" should match a fresh hash of "bcd".
        let mut rh3 = RollingHash::default_params(3);
        rh3.init(b"bcd");
        assert_eq!(rh2.hash(), rh3.hash());
        assert_ne!(rh2.hash(), h_abc);
    }

    #[test]
    fn test_rolling_hash_different_strings() {
        let mut rh1 = RollingHash::default_params(4);
        rh1.init(b"rust");
        let mut rh2 = RollingHash::default_params(4);
        rh2.init(b"java");
        assert_ne!(rh1.hash(), rh2.hash());
    }

    // ---- DoubleRollingHash --------------------------------------------------

    #[test]
    fn test_double_rolling_hash_roll() {
        let mut drh = DoubleRollingHash::new(3);
        drh.init(b"abc");
        let h_abc = drh.hash();
        drh.roll(b'a', b'd');
        assert_ne!(drh.hash(), h_abc);
    }

    #[test]
    fn test_double_rolling_hash_same_content() {
        let mut drh1 = DoubleRollingHash::new(3);
        drh1.init(b"abc");
        let mut drh2 = DoubleRollingHash::new(3);
        drh2.init(b"abc");
        assert_eq!(drh1.hash(), drh2.hash());
    }

    // ---- Rabin-Karp search --------------------------------------------------

    #[test]
    fn test_rabin_karp_basic() {
        let pos = rabin_karp_search(b"ababab", b"ab");
        assert_eq!(pos, vec![0, 2, 4]);
    }

    #[test]
    fn test_rabin_karp_no_match() {
        let pos = rabin_karp_search(b"hello world", b"xyz");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_rabin_karp_pattern_equals_text() {
        let pos = rabin_karp_search(b"abc", b"abc");
        assert_eq!(pos, vec![0]);
    }

    #[test]
    fn test_rabin_karp_pattern_longer() {
        let pos = rabin_karp_search(b"ab", b"abc");
        assert!(pos.is_empty());
    }

    #[test]
    fn test_rabin_karp_overlapping() {
        let pos = rabin_karp_search(b"aaa", b"aa");
        assert_eq!(pos, vec![0, 1]);
    }

    // ---- Unique substrings --------------------------------------------------

    #[test]
    fn test_find_unique_substrings_basic() {
        let unique = find_unique_substrings(b"abab", 2);
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(b"ab".as_ref()));
        assert!(unique.contains(b"ba".as_ref()));
    }

    #[test]
    fn test_find_unique_substrings_k_larger_than_text() {
        let unique = find_unique_substrings(b"ab", 5);
        assert!(unique.is_empty());
    }

    // ---- Distinct k-mers ---------------------------------------------------

    #[test]
    fn test_count_distinct_kmers_basic() {
        assert_eq!(count_distinct_kmers(b"abab", 2), 2);
    }

    #[test]
    fn test_count_distinct_kmers_all_same() {
        assert_eq!(count_distinct_kmers(b"aaaa", 2), 1);
    }

    #[test]
    fn test_count_distinct_kmers_all_different() {
        // "abcd" has 3 di-grams: "ab", "bc", "cd" — all distinct
        assert_eq!(count_distinct_kmers(b"abcd", 2), 3);
    }

    // ---- Levenshtein distance ----------------------------------------------

    #[test]
    fn test_levenshtein_kitten_sitting() {
        assert_eq!(levenshtein_distance(b"kitten", b"sitting"), 3);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein_distance(b"", b"abc"), 3);
        assert_eq!(levenshtein_distance(b"abc", b""), 3);
        assert_eq!(levenshtein_distance(b"", b""), 0);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance(b"hello", b"hello"), 0);
    }

    #[test]
    fn test_levenshtein_single_substitution() {
        assert_eq!(levenshtein_distance(b"abc", b"axc"), 1);
    }

    // ---- LCS ----------------------------------------------------------------

    #[test]
    fn test_lcs_length_basic() {
        assert_eq!(lcs_length(b"ABCBDAB", b"BDCAB"), 4);
    }

    #[test]
    fn test_lcs_length_identical() {
        assert_eq!(lcs_length(b"abc", b"abc"), 3);
    }

    #[test]
    fn test_lcs_length_disjoint() {
        assert_eq!(lcs_length(b"abc", b"xyz"), 0);
    }

    #[test]
    fn test_lcs_sequence_length() {
        let seq = lcs_sequence(b"ABCBDAB", b"BDCAB");
        assert_eq!(seq.len(), 4);
    }

    #[test]
    fn test_lcs_sequence_valid_subsequence() {
        let a = b"ABCBDAB";
        let b = b"BDCAB";
        let seq = lcs_sequence(a, b);
        // Check seq is actually a subsequence of a.
        let mut ai = 0usize;
        for &c in &seq {
            while ai < a.len() && a[ai] != c {
                ai += 1;
            }
            assert!(ai < a.len(), "LCS element not found in a");
            ai += 1;
        }
    }

    // ---- Hash utilities -----------------------------------------------------

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_hash(b"hello"), fnv1a_hash(b"hello"));
    }

    #[test]
    fn test_fnv1a_different_strings() {
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    #[test]
    fn test_djb2_deterministic() {
        assert_eq!(djb2_hash(b"hello"), djb2_hash(b"hello"));
    }

    #[test]
    fn test_djb2_different_strings() {
        assert_ne!(djb2_hash(b"foo"), djb2_hash(b"bar"));
    }

    #[test]
    fn test_fnv1a_known_value() {
        // FNV-1a of empty string is the offset basis.
        assert_eq!(fnv1a_hash(b""), 14_695_981_039_346_656_037u64);
    }
}
