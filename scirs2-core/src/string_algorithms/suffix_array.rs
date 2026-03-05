//! Suffix array construction, LCP array (Kasai's algorithm), BWT encode/decode,
//! longest repeated substring, and longest common substring.
//!
//! Construction uses the O(n log n) prefix-doubling (Manber-Myers) algorithm.
//! LCP array is built with Kasai's O(n) algorithm.

/// Suffix array together with the inverse suffix array and LCP array.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::SuffixArray;
///
/// let sa = SuffixArray::build(b"banana");
/// // Find all occurrences of "ana"
/// let positions = sa.find_all(b"banana", b"ana");
/// assert_eq!(positions.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SuffixArray {
    /// `sa[i]` = starting index of the i-th lexicographically smallest suffix.
    pub sa: Vec<usize>,
    /// `isa[j]` = rank (position in the sorted order) of the suffix starting at j.
    pub isa: Vec<usize>,
    /// `lcp[i]` = length of the longest common prefix between `sa[i-1]` and `sa[i]`.
    /// `lcp[0]` is conventionally 0.
    pub lcp: Vec<usize>,
    text_len: usize,
}

impl SuffixArray {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a suffix array from a byte slice using O(n log n) prefix doubling.
    pub fn build(text: &[u8]) -> Self {
        let n = text.len();
        if n == 0 {
            return SuffixArray {
                sa: Vec::new(),
                isa: Vec::new(),
                lcp: Vec::new(),
                text_len: 0,
            };
        }

        // Initial rank based on the first byte of each suffix.
        let mut sa: Vec<usize> = (0..n).collect();
        let mut rank: Vec<i64> = text.iter().map(|&b| b as i64).collect();

        // Stable sort by initial rank.
        sa.sort_by_key(|&i| rank[i]);

        // Prefix-doubling: double the comparison key length each round.
        let mut gap = 1usize;
        while gap < n {
            let cur_rank = rank.clone();

            // Comparator: (rank[i], rank[i+gap]) vs (rank[j], rank[j+gap]).
            let second_rank = |i: usize| -> i64 {
                if i + gap < n {
                    cur_rank[i + gap]
                } else {
                    -1
                }
            };

            sa.sort_by(|&a, &b| {
                let ka = (cur_rank[a], second_rank(a));
                let kb = (cur_rank[b], second_rank(b));
                ka.cmp(&kb)
            });

            // Rebuild ranks from the sorted order.
            let mut new_rank = vec![0i64; n];
            new_rank[sa[0]] = 0;
            for i in 1..n {
                let prev = sa[i - 1];
                let cur = sa[i];
                let same = cur_rank[prev] == cur_rank[cur]
                    && second_rank(prev) == second_rank(cur);
                new_rank[cur] = new_rank[prev] + if same { 0 } else { 1 };
            }
            rank = new_rank;

            // Early exit when all ranks are unique.
            if rank[sa[n - 1]] == (n as i64 - 1) {
                break;
            }

            gap <<= 1;
        }

        // Build inverse suffix array.
        let mut isa = vec![0usize; n];
        for (i, &s) in sa.iter().enumerate() {
            isa[s] = i;
        }

        // Build LCP array with Kasai's algorithm.
        let lcp = Self::build_lcp(text, &sa, &isa);

        SuffixArray {
            sa,
            isa,
            lcp,
            text_len: n,
        }
    }

    /// Build a suffix array from a UTF-8 string.
    pub fn build_str(text: &str) -> Self {
        Self::build(text.as_bytes())
    }

    /// Kasai's O(n) LCP array construction.
    fn build_lcp(text: &[u8], sa: &[usize], isa: &[usize]) -> Vec<usize> {
        let n = text.len();
        let mut lcp = vec![0usize; n];
        let mut h = 0usize;
        for i in 0..n {
            if isa[i] > 0 {
                let j = sa[isa[i] - 1];
                while i + h < n && j + h < n && text[i + h] == text[j + h] {
                    h += 1;
                }
                lcp[isa[i]] = h;
                if h > 0 {
                    h -= 1;
                }
            }
        }
        lcp
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Binary-search for `pattern` in `text` using the precomputed suffix array.
    ///
    /// Returns `Some((lo, hi))` where `[lo, hi)` is the half-open range in `sa`
    /// that covers all occurrences, or `None` if the pattern is absent.
    pub fn search(&self, text: &[u8], pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() || self.text_len == 0 {
            return None;
        }
        let n = self.text_len;
        let m = pattern.len();

        // Lower bound: first suffix >= pattern.
        let lo = {
            let mut low = 0usize;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                let start = self.sa[mid];
                let end = (start + m).min(n);
                if text[start..end] < *pattern {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };

        // Upper bound: first suffix > pattern.
        let hi = {
            let mut low = lo;
            let mut high = n;
            while low < high {
                let mid = low + (high - low) / 2;
                let start = self.sa[mid];
                let end = (start + m).min(n);
                if text[start..end] <= *pattern {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };

        if lo < hi {
            Some((lo, hi))
        } else {
            None
        }
    }

    /// Count the number of times `pattern` appears in `text`.
    pub fn count_occurrences(&self, text: &[u8], pattern: &[u8]) -> usize {
        self.search(text, pattern)
            .map(|(lo, hi)| hi - lo)
            .unwrap_or(0)
    }

    /// Find all starting positions of `pattern` in `text` (unsorted order).
    pub fn find_all(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        match self.search(text, pattern) {
            Some((lo, hi)) => {
                let mut positions: Vec<usize> = self.sa[lo..hi].to_vec();
                positions.sort_unstable();
                positions
            }
            None => Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Derived queries
    // -----------------------------------------------------------------------

    /// Return the longest substring that appears at least twice in `text`.
    ///
    /// The returned slice is a view into `text`.
    pub fn longest_repeated_substring<'a>(&self, text: &'a [u8]) -> &'a [u8] {
        if self.lcp.is_empty() {
            return &text[0..0];
        }
        let (max_lcp, max_i) = self
            .lcp
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, &v)| (v, i))
            .unwrap_or((0, 0));

        if max_lcp == 0 {
            return &text[0..0];
        }
        let start = self.sa[max_i];
        &text[start..start + max_lcp]
    }

    /// Find the longest common substring shared between `text1` and `text2`.
    ///
    /// Uses a generalised suffix array constructed over the concatenation
    /// `text1 ++ separator ++ text2` (separator byte 0x01, below printable
    /// ASCII, ensuring no spurious matches across the boundary).
    ///
    /// The returned slice borrows from `text1`.
    pub fn longest_common_substring<'a>(text1: &'a [u8], text2: &[u8]) -> &'a [u8] {
        if text1.is_empty() || text2.is_empty() {
            return &text1[0..0];
        }

        // Concatenate with a separator byte that cannot appear in either string
        // in a context that bridges the two texts.
        let sep = 0x01u8;
        let mut combined = Vec::with_capacity(text1.len() + 1 + text2.len());
        combined.extend_from_slice(text1);
        combined.push(sep);
        combined.extend_from_slice(text2);

        let n1 = text1.len();
        let sa = SuffixArray::build(&combined);

        let mut best_len = 0usize;
        let mut best_start = 0usize;

        // Scan adjacent pairs in the suffix array.  A pair is "cross-boundary"
        // when one suffix starts in text1 (< n1) and the other in text2 (> n1).
        let n = sa.sa.len();
        for i in 1..n {
            let lcp_val = sa.lcp[i];
            if lcp_val == 0 {
                continue;
            }
            let a = sa.sa[i - 1];
            let b = sa.sa[i];
            let a_in_t1 = a < n1;
            let b_in_t1 = b < n1;
            if a_in_t1 != b_in_t1 && lcp_val > best_len {
                // Make sure the common prefix doesn't cross the separator.
                let start1 = if a_in_t1 { a } else { b };
                if start1 + lcp_val <= n1 {
                    best_len = lcp_val;
                    best_start = start1;
                }
            }
        }

        &text1[best_start..best_start + best_len]
    }
}

// ---------------------------------------------------------------------------
// Burrows-Wheeler Transform
// ---------------------------------------------------------------------------

/// Encode `text` with the Burrows-Wheeler Transform.
///
/// Returns `(bwt_string, original_row)` where `original_row` is the row in
/// the sorted rotation matrix that corresponds to the original string — needed
/// for decoding.
///
/// # Example
///
/// ```rust
/// use scirs2_core::string_algorithms::{bwt_encode, bwt_decode};
///
/// let (bwt, pos) = bwt_encode(b"banana");
/// let decoded = bwt_decode(&bwt, pos);
/// assert_eq!(decoded, b"banana");
/// ```
pub fn bwt_encode(text: &[u8]) -> (Vec<u8>, usize) {
    let n = text.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Build suffix array and derive the BWT directly.
    let sa = SuffixArray::build(text);
    let mut bwt = Vec::with_capacity(n);
    let mut original_pos = 0usize;

    for (i, &start) in sa.sa.iter().enumerate() {
        if start == 0 {
            bwt.push(text[n - 1]);
            original_pos = i;
        } else {
            bwt.push(text[start - 1]);
        }
    }

    (bwt, original_pos)
}

/// Decode a Burrows-Wheeler Transform.
///
/// `bwt` is the output of [`bwt_encode`] and `original_pos` is the index
/// returned by that function.
pub fn bwt_decode(bwt: &[u8], original_pos: usize) -> Vec<u8> {
    let n = bwt.len();
    if n == 0 {
        return Vec::new();
    }

    // Build first column F (sorted BWT) and the LF-mapping.
    let mut sorted = bwt.to_vec();
    sorted.sort_unstable();

    // Count occurrences of each byte in BWT.
    let mut count = [0usize; 256];
    for &b in bwt {
        count[b as usize] += 1;
    }

    // Compute the starting position of each byte in F.
    let mut first_occ = [0usize; 256];
    let mut total = 0usize;
    for (b, c) in count.iter().enumerate() {
        first_occ[b] = total;
        total += c;
    }

    // Build the LF-mapping: LF[i] is the row in the matrix that is reached
    // from row i by prepending bwt[i].
    let mut byte_rank = [0usize; 256]; // running count per byte in BWT
    let mut lf = vec![0usize; n];
    for (i, &b) in bwt.iter().enumerate() {
        lf[i] = first_occ[b as usize] + byte_rank[b as usize];
        byte_rank[b as usize] += 1;
    }

    // Reconstruct original string by following LF-mapping backwards.
    let mut result = vec![0u8; n];
    let mut row = original_pos;
    for slot in result.iter_mut().rev() {
        *slot = bwt[row];
        row = lf[row];
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Suffix array -------------------------------------------------------

    #[test]
    fn test_sa_build_banana() {
        let sa = SuffixArray::build(b"banana");
        // The sorted suffixes of "banana" are:
        // a, ana, anana, banana, na, nana
        // => starting positions: 5, 3, 1, 0, 4, 2
        assert_eq!(sa.sa, vec![5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_sa_inverse() {
        let sa = SuffixArray::build(b"banana");
        for (i, &s) in sa.sa.iter().enumerate() {
            assert_eq!(sa.isa[s], i);
        }
    }

    #[test]
    fn test_sa_lcp_banana() {
        let sa = SuffixArray::build(b"banana");
        // lcp[0]=0, then between consecutive sorted suffixes:
        // (5="a", 3="ana") => 1; (3="ana", 1="anana") => 3;
        // (1="anana", 0="banana") => 0; (0="banana", 4="na") => 0;
        // (4="na", 2="nana") => 2
        assert_eq!(sa.lcp[0], 0);
        assert_eq!(sa.lcp[1], 1); // "a" vs "ana"
        assert_eq!(sa.lcp[2], 3); // "ana" vs "anana"
        assert_eq!(sa.lcp[3], 0); // "anana" vs "banana"
        assert_eq!(sa.lcp[4], 0); // "banana" vs "na"
        assert_eq!(sa.lcp[5], 2); // "na" vs "nana"
    }

    #[test]
    fn test_sa_search_found() {
        let text = b"banana";
        let sa = SuffixArray::build(text);
        let range = sa.search(text, b"ana");
        assert!(range.is_some());
    }

    #[test]
    fn test_sa_search_not_found() {
        let text = b"banana";
        let sa = SuffixArray::build(text);
        assert!(sa.search(text, b"xyz").is_none());
    }

    #[test]
    fn test_sa_count_occurrences() {
        let text = b"banana";
        let sa = SuffixArray::build(text);
        assert_eq!(sa.count_occurrences(text, b"ana"), 2);
        assert_eq!(sa.count_occurrences(text, b"na"), 2);
        assert_eq!(sa.count_occurrences(text, b"banana"), 1);
    }

    #[test]
    fn test_sa_find_all() {
        let text = b"banana";
        let sa = SuffixArray::build(text);
        let mut pos = sa.find_all(text, b"ana");
        pos.sort_unstable();
        assert_eq!(pos, vec![1, 3]);
    }

    #[test]
    fn test_sa_empty_text() {
        let sa = SuffixArray::build(b"");
        assert!(sa.sa.is_empty());
    }

    #[test]
    fn test_sa_single_char() {
        let text = b"aaaa";
        let sa = SuffixArray::build(text);
        assert_eq!(sa.count_occurrences(text, b"a"), 4);
        assert_eq!(sa.count_occurrences(text, b"aa"), 3);
    }

    #[test]
    fn test_sa_longest_repeated_substring() {
        let text = b"banana";
        let sa = SuffixArray::build(text);
        let lrs = sa.longest_repeated_substring(text);
        // Maximum LCP is 3 ("ana"), repeated at positions 1 and 3.
        assert_eq!(lrs, b"ana");
    }

    #[test]
    fn test_sa_longest_common_substring() {
        let lcs = SuffixArray::longest_common_substring(b"xabcy", b"zabc");
        assert_eq!(lcs, b"abc");
    }

    #[test]
    fn test_sa_lcs_empty() {
        let lcs = SuffixArray::longest_common_substring(b"abc", b"xyz");
        assert_eq!(lcs, b"");
    }

    #[test]
    fn test_sa_lcs_full_overlap() {
        let text = b"hello";
        let lcs = SuffixArray::longest_common_substring(text, text);
        assert_eq!(lcs, b"hello");
    }

    // ---- BWT ----------------------------------------------------------------

    #[test]
    fn test_bwt_encode_decode_banana() {
        let (bwt, pos) = bwt_encode(b"banana");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"banana");
    }

    #[test]
    fn test_bwt_encode_decode_empty() {
        let (bwt, pos) = bwt_encode(b"");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"");
    }

    #[test]
    fn test_bwt_encode_decode_single() {
        let (bwt, pos) = bwt_encode(b"a");
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, b"a");
    }

    #[test]
    fn test_bwt_encode_decode_repeated() {
        let original = b"aaaaaa";
        let (bwt, pos) = bwt_encode(original);
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, original.as_ref());
    }

    #[test]
    fn test_bwt_encode_decode_general() {
        let original = b"mississippi";
        let (bwt, pos) = bwt_encode(original);
        let decoded = bwt_decode(&bwt, pos);
        assert_eq!(decoded, original.as_ref());
    }
}
